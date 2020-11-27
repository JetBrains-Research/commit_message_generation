import pytorch_lightning as pl

import torch
from torch import nn

from transformers import EncoderDecoderModel, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup

import wandb

from metrics import AccuracyMetric
from metrics import BleuMetric


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 reduction: str,
                 model_name_or_path: str,
                 tokenizer: RobertaTokenizer,
                 num_epochs: int,
                 num_batches: int,
                 **kwargs):
        super().__init__()

        self._tokenizer = tokenizer
        self._num_epochs = num_epochs
        self._num_batches = num_batches

        self.save_hyperparameters()

        self.learning_rate = learning_rate

        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_name_or_path, model_name_or_path)

        self.loss = nn.CrossEntropyLoss(reduction=reduction, ignore_index=self.pad_token_id)

        self.accuracy = AccuracyMetric(self.pad_token_id)
        self.bleu = BleuMetric()

    def forward(self, batch):
        return self.model(input_ids=batch[0]['input_ids'],
                          attention_mask=batch[0]['attention_mask'],
                          decoder_input_ids=batch[1]['input_ids'],
                          decoder_attention_mask=batch[1]['attention_mask'],
                          return_dict=True)

    def generate(self, batch):
        return self.model.generate(input_ids=batch[0]['input_ids'],
                                   max_length=batch[1]['input_ids'].shape[1],
                                   min_length=batch[1]['input_ids'].shape[1],
                                   decoder_start_token_id=self.bos_token_id,
                                   do_sample=True,
                                   top_p=0.92,
                                   top_k=0,
                                   pad_token_id=self.pad_token_id,
                                   bos_token_id=self.bos_token_id,
                                   eos_token_id=self.eos_token_id)

    def training_step(self, batch, batch_idx):
        logits = self(batch)['logits']
        train_loss = self.loss(logits.view(-1, logits.size(-1)), batch[1]['input_ids'].view(-1))

        # log train examples on first batch in epoch
        if self.global_step % self._num_batches == 0:
            gen_sequence = self.generate(batch)

            _, _, table = self.compute_metrics(batch[0]['input_ids'], gen_sequence, batch[1]['input_ids'])
        else:
            table = None

        self.logger.experiment.log({"train_loss_step": train_loss})
        return {"loss": train_loss, "examples": table}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean().item()
        tables = [x["examples"] for x in outputs if x["examples"] is not None]
        try:
            self.logger.experiment.log({"train_examples": tables[0],
                                        "train_loss_epoch": train_loss_mean})
        except IndexError:
            self.logger.experiment.log({"train_loss_epoch": train_loss_mean})

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch)['logits']
        val_loss = self.loss(logits.view(-1, logits.size(-1)), batch[1]['input_ids'].view(-1))

        gen_sequence = self.generate(batch)

        acc, bleu, table = self.compute_metrics(batch[0]['input_ids'], gen_sequence, batch[1]['input_ids'])

        return {"val_loss": val_loss, "val_accuracy": acc, "val_bleu": bleu, "examples": table}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().item()
        val_acc_mean = torch.stack([x["val_accuracy"] for x in outputs]).mean().item()
        val_bleu_mean = torch.stack([x["val_bleu"] for x in outputs]).mean().item()
        self.logger.experiment.log({"val_examples": outputs[0]["examples"],
                                    "val_accuracy": val_acc_mean,
                                    "val_bleu": val_bleu_mean,
                                    "val_loss": val_loss_mean})

    def test_step(self, batch, batch_idx):
        gen_sequence = self.generate(batch)

        acc, bleu, table = self.compute_metrics(batch[0]['input_ids'], gen_sequence, batch[1]['input_ids'])
        return {"test_accuracy": acc, "test_bleu": bleu, "examples": table}

    def test_epoch_end(self, outputs):
        test_acc_mean = torch.stack([x["test_accuracy"] for x in outputs]).mean().item()
        test_bleu_mean = torch.stack([x["test_bleu"] for x in outputs]).mean().item()
        self.logger.experiment.log({"test_examples": outputs[0]["examples"],
                                    "test_accuracy": test_acc_mean,
                                    "test_bleu": test_bleu_mean})

    def compute_metrics(self, source, generated, target, n_examples=10):
        # compute accuracy with tensors
        acc = self.accuracy(generated, target)

        # compute BLEU with decoded strings
        targets = self._tokenizer.batch_decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        preds = self._tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        bleu = self.bleu(preds, targets)

        # log a little table with examples
        table = wandb.Table(columns=["Source", "Predicted", "Target"])
        srcs = self._tokenizer.batch_decode(source, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i in range(n_examples):
            try:
                table.add_data(srcs[i], preds[i], targets[i])
            except IndexError:
                break
        return acc, bleu, table

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer, self._num_batches//2, self._num_epochs * self._num_batches),
                     'name': 'learning_rate',
                     'interval': 'step',
                     'frequency': 1}
        return [optimizer], [scheduler]
