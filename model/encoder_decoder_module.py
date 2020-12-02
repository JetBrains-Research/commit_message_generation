import pytorch_lightning as pl

import torch
import torch.nn.functional as F

from transformers import EncoderDecoderModel, RobertaTokenizer, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

import wandb

from metrics import AccuracyMetric
from metrics import BleuMetric


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 encoder_name_or_path: str,
                 decoder_name_or_path: str,
                 src_tokenizer: RobertaTokenizer,
                 trg_tokenizer: GPT2Tokenizer,
                 num_epochs: int,
                 num_batches: int,
                 **kwargs):
        super().__init__()

        self._src_tokenizer = src_tokenizer
        self._trg_tokenizer = trg_tokenizer
        self._num_epochs = num_epochs
        self._num_batches = num_batches

        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.pad_token_id = src_tokenizer.pad_token_id
        self.bos_token_id = src_tokenizer.bos_token_id
        self.eos_token_id = src_tokenizer.eos_token_id

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name_or_path, decoder_name_or_path)

        # do not tie output embeddings to input embeddings
        self.model.config.tie_word_embeddings = False

        # set decoding params
        self.model.config.decoder_start_token_id = self._trg_tokenizer.bos_token_id
        self.model.config.bos_token_id = self._trg_tokenizer.bos_token_id
        self.model.config.eos_token_id = self._trg_tokenizer.eos_token_id
        self.model.config.pad_token_id = self._trg_tokenizer.pad_token_id
        self.model.config.max_length = 50
        self.model.config.min_length = 5
        self.model.config.no_repeat_ngram_size = 3
        self.model.early_stopping = True
        self.model.length_penalty = 1.5
        self.model.num_beams = 4

        print("\n==CONFIG==\n")
        print(self.model.config)
        print()

        self.accuracy = AccuracyMetric(self.pad_token_id)
        self.bleu = BleuMetric()

    def forward(self, batch):

        # transformers assume pad indices to be -100
        # gpt2 has no pad tokens so use attention mask
        return self.model(input_ids=batch[0]['input_ids'],
                          attention_mask=batch[0]['attention_mask'],
                          decoder_input_ids=batch[1]['input_ids'],
                          decoder_attention_mask=batch[1]['attention_mask'],
                          labels=batch[1]['input_ids'].where(batch[1]['attention_mask'].type(torch.ByteTensor).to(self.device),
                                                             torch.tensor(-100, device=self.device)))

    def generate(self, batch):
        return self.model.generate(input_ids=batch[0]['input_ids'],
                                   attention_mask=batch[0]['attention_mask'])

    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)[:2]

        # log train examples on every 1000th batch in epoch
        if self.global_step % 1000 == 0:
            with torch.no_grad():
                gen_sequence = self.generate(batch)
                _, _, table = self.compute_metrics(batch[0]['input_ids'], gen_sequence, batch[1]['input_ids'])
        else:
            table = None

        self.logger.experiment.log({"train_loss_step": loss})
        return {"loss": loss, "examples": table}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        tables = [x["examples"] for x in outputs if x["examples"] is not None]
        try:
            self.logger.experiment.log({"train_examples": tables[0],
                                        "train_loss_epoch": train_loss_mean})
        except IndexError:
            self.logger.experiment.log({"train_loss_epoch": train_loss_mean})

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits = self(batch)[:2]

        gen_sequence = self.generate(batch)

        acc, bleu, table = self.compute_metrics(batch[0]['input_ids'], gen_sequence, batch[1]['input_ids'])

        return {"val_loss": loss, "val_accuracy": acc, "val_bleu": bleu, "examples": table}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc_mean = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        val_bleu_mean = torch.stack([x["val_bleu"] for x in outputs]).mean()
        self.logger.experiment.log({"val_examples": outputs[0]["examples"],
                                    "val_accuracy": val_acc_mean,
                                    "val_bleu": val_bleu_mean,
                                    "val_loss": val_loss_mean})

    def test_step(self, batch, batch_idx):
        gen_sequence = self.generate(batch)
        acc, bleu, table = self.compute_metrics(batch[0]['input_ids'], gen_sequence, batch[1]['input_ids'])
        return {"test_accuracy": acc, "test_bleu": bleu, "examples": table}

    def test_epoch_end(self, outputs):
        test_acc_mean = torch.stack([x["test_accuracy"] for x in outputs]).mean()
        test_bleu_mean = torch.stack([x["test_bleu"] for x in outputs]).mean()
        self.logger.experiment.log({"test_examples": outputs[0]["examples"],
                                    "test_accuracy": test_acc_mean,
                                    "test_bleu": test_bleu_mean})

    def compute_metrics(self, source, generated, target, n_examples=10):
        if target.shape[1] > generated.shape[1]:
            # pad generated tokens to match sequence length dimension with target
            generated = F.pad(input=generated, pad=(0, target.shape[1] - generated.shape[1], 0, 0), mode='constant',
                              value=self.pad_token_id)
        elif generated.shape[1] > target.shape[1]:
            # pad target tokens to match sequence length dimension with generated
            target = F.pad(input=target, pad=(0, generated.shape[1] - target.shape[1], 0, 0), mode='constant',
                           value=self.pad_token_id)
        # compute accuracy with tensors
        acc = self.accuracy(generated, target)

        # compute BLEU with decoded strings
        targets = self._trg_tokenizer.batch_decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        preds = self._trg_tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
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
        scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer, self._num_batches * 8,
                                                                  self._num_epochs * self._num_batches),
                     'interval': 'step',
                     'frequency': 1}
        return [optimizer], [scheduler]
