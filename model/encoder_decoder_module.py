import pytorch_lightning as pl

import torch
import torch.nn.functional as F

from transformers import EncoderDecoderModel, RobertaModel, GPT2LMHeadModel, GPT2Config,\
    RobertaTokenizer, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

import wandb

from metrics import AccuracyMetric
from metrics import BleuMetric


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 encoder_name_or_path: str,
                 decoder_name_or_path: str,
                 unfreeze_after: int,
                 src_tokenizer: RobertaTokenizer,
                 trg_tokenizer: GPT2Tokenizer,
                 num_epochs: int,
                 num_batches: int,
                 **kwargs):
        super().__init__()

        self._unfreeze_after = unfreeze_after
        self._src_tokenizer = src_tokenizer
        self._trg_tokenizer = trg_tokenizer
        self._num_epochs = num_epochs
        self._num_batches = num_batches

        self.save_hyperparameters()

        self.learning_rate = learning_rate

        # use RoBERTa with resized embeddings as encoder
        encoder = RobertaModel.from_pretrained(encoder_name_or_path)
        # resize embeddings to match special token
        encoder.resize_token_embeddings(len(self._src_tokenizer))
        encoder.config.type_vocab_size = 2
        encoder.embeddings.token_type_embeddings = torch.nn.Embedding.from_pretrained(
                                         torch.cat((encoder.embeddings.token_type_embeddings.weight,
                                                    encoder.embeddings.token_type_embeddings.weight), dim=0))
        # use GPT-2 as decoder
        config = GPT2Config.from_pretrained(decoder_name_or_path)
        config.is_decoder = True
        config.add_cross_attention = True
        gpt = GPT2LMHeadModel.from_pretrained(decoder_name_or_path, config=config)

        self.model = EncoderDecoderModel(encoder=encoder, decoder=gpt)

        # cache is currently not supported by EncoderDecoder framework
        self.model.decoder.config.use_cache = False

        # do not tie output embeddings to input embeddings
        self.model.config.tie_word_embeddings = False

        # set decoding params
        self.model.config.decoder_start_token_id = self._trg_tokenizer.bos_token_id
        self.model.config.bos_token_id = self._trg_tokenizer.bos_token_id
        self.model.config.eos_token_id = self._trg_tokenizer.eos_token_id
        self.model.config.pad_token_id = self._trg_tokenizer.pad_token_id
        self.model.config.max_length = 30
        self.model.config.min_length = 2
        self.model.config.no_repeat_ngram_size = 4
        self.model.config.early_stopping = True
        self.model.config.num_beams = 4

        print("\n====MODEL CONFIG====\n")
        print(self.model.config)
        print()

        self.accuracy = AccuracyMetric(self._trg_tokenizer.pad_token_id)
        self.bleu = BleuMetric()

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def on_train_epoch_start(self) -> None:
        # freeze codebert after several epochs
        if self.current_epoch == self._unfreeze_after:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

    def forward(self, batch):
        self.examples_count += len(batch[0])

        encoder_outputs = self.model.encoder(
            input_ids=batch[0],
            attention_mask=batch[1],
            token_type_ids=batch[2])

        # transformers assume pad indices to be -100
        # gpt2 has no pad tokens so use attention mask
        return self.model(encoder_outputs=encoder_outputs,
                          decoder_input_ids=batch[3],
                          decoder_attention_mask=batch[4],
                          labels=batch[3].where(batch[4].type(torch.ByteTensor).to(self.device),
                                                torch.tensor(-100, device=self.device)))

    def generate(self, batch):
        return self.model.generate(input_ids=batch[0],
                                   attention_mask=batch[1])

    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)[:2]

        # log train examples on every 1000th batch in epoch
        if self.global_step % 1000 == 0:
            with torch.no_grad():
                gen_sequence = self.generate(batch)
                _, _, table = self.compute_metrics(batch[0], gen_sequence, batch[3])
        else:
            table = None

        self.logger.experiment.log({"train_loss_step": loss}, step=self.examples_count)
        return {"loss": loss, "examples": table}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        tables = [x["examples"] for x in outputs if x["examples"] is not None]
        try:
            self.logger.experiment.log({"train_examples": tables[0],
                                        "train_loss_epoch": train_loss_mean}, step=self.examples_count)
        except IndexError:
            self.logger.experiment.log({"train_loss_epoch": train_loss_mean}, step=self.examples_count)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits = self(batch)[:2]

        gen_sequence = self.generate(batch)

        acc, bleu, table = self.compute_metrics(batch[0], gen_sequence, batch[3])

        return {"val_loss": loss, "val_accuracy": acc, "val_bleu": bleu, "examples": table}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        val_bleu_mean = torch.stack([x["val_bleu"] for x in outputs]).mean()
        self.logger.experiment.log({"val_examples": outputs[0]["examples"],
                                    "val_accuracy": val_acc_mean,
                                    "val_bleu": val_bleu_mean,
                                    "val_loss": val_loss_mean}, step=self.examples_count)

    def test_step(self, batch, batch_idx):
        gen_sequence = self.generate(batch)
        acc, bleu, table = self.compute_metrics(batch[0], gen_sequence, batch[3])
        return {"test_accuracy": acc, "test_bleu": bleu, "examples": table}

    def test_epoch_end(self, outputs):
        test_acc_mean = torch.stack([x["test_accuracy"] for x in outputs]).mean()
        test_bleu_mean = torch.stack([x["test_bleu"] for x in outputs]).mean()
        self.logger.experiment.log({"test_examples": outputs[0]["examples"],
                                    "test_accuracy": test_acc_mean,
                                    "test_bleu": test_bleu_mean}, step=self.examples_count)

    def compute_metrics(self, source, generated, target, n_examples=10):
        if target.shape[1] > generated.shape[1]:
            # pad generated tokens to match sequence length dimension with target
            generated = F.pad(input=generated, pad=(0, target.shape[1] - generated.shape[1], 0, 0), mode='constant',
                              value=self._trg_tokenizer.pad_token_id)
        elif generated.shape[1] > target.shape[1]:
            # pad target tokens to match sequence length dimension with generated
            target = F.pad(input=target, pad=(0, generated.shape[1] - target.shape[1], 0, 0), mode='constant',
                           value=self._trg_tokenizer.pad_token_id)
        # compute accuracy with tensors
        acc = self.accuracy(generated, target)

        # compute BLEU with decoded strings
        targets = self._trg_tokenizer.batch_decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        preds = self._trg_tokenizer.batch_decode(generated, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
        bleu = self.bleu(preds, targets)

        # log a little table with examples
        table = wandb.Table(columns=["Source", "Predicted", "Target"])
        srcs = self._src_tokenizer.batch_decode(source, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i in range(n_examples):
            try:
                table.add_data(srcs[i], preds[i], targets[i])
            except IndexError:
                break
        return acc, bleu, table

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer, self._num_batches * 5,
                                                                  self._num_epochs * self._num_batches),
                     'interval': 'step',
                     'frequency': 1}
        return [optimizer], [scheduler]
