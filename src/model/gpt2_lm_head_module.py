import numpy as np
import pytorch_lightning as pl
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

import nltk

nltk.download("wordnet")


class GPT2LMHeadModule(pl.LightningModule):
    def __init__(
        self,
        decoder_name_or_path: str,
        tokenizer: GPT2Tokenizer,
        learning_rate: float,
        num_epochs: int,
        num_batches: int,
        num_gpus: int,
        **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self._tokenizer = tokenizer
        self.save_hyperparameters()

        self._num_epochs = num_epochs
        self._num_batches = num_batches
        self._num_gpus = num_gpus

        # use pretrained GPT-2 as decoder
        self.model = GPT2LMHeadModel.from_pretrained(decoder_name_or_path)

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def forward(self, batch):
        self.examples_count += len(batch["msg_input_ids"])
        return self.model(
            input_ids=batch["msg_input_ids"], attention_mask=batch["msg_attention_mask"], labels=batch["msg_labels"]
        )

    def training_step(self, batch, batch_idx):
        self.examples_count += len(batch["diff_input_ids"])
        loss, logits = self(batch)[:2]
        self.logger.experiment.log({"train_loss_step": loss}, step=self.examples_count)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.log({"train_loss_epoch": train_loss_mean}, step=self.examples_count)

    def next_token_metrics_step(self, batch):
        loss, scores = self(batch)[:2]
        return {"loss": loss}

    def next_token_metrics_epoch_end(self, outputs, stage):
        """
        Logic for validation & testing epoch end:
        1) Calculate accuracy@1, accuracy@5, MRR@5
        2) (in val stage only) Aggregate loss and log metric(s) for ModelCheckpoint
        3) Log everything to wandb
        """
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        metrics = {f"{stage}_loss_epoch": loss}
        if stage == "val":
            self.log("val_loss_epoch", metrics["val_loss_epoch"], on_step=False, on_epoch=True, prog_bar=True,
                     logger=False)
        self.logger.experiment.log(metrics, step=self.examples_count)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.next_token_metrics_step(batch)

    def validation_epoch_end(self, outputs):
        self.next_token_metrics_epoch_end(outputs, "val")

    def test_step(self, batch, batch_idx):
        return self.next_token_metrics_step(batch)

    def test_epoch_end(self, outputs):
        self.next_token_metrics_epoch_end(outputs, "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer, 4000 // self._num_gpus, self._num_epochs * self._num_batches
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
