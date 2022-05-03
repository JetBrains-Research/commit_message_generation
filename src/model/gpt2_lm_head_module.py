from collections import defaultdict

import pytorch_lightning as pl
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

from src.utils import EvaluationMetrics, PrefixAllowedTokens

import wandb
import nltk
import pandas as pd

nltk.download("wordnet")


class GPT2LMHeadModule(pl.LightningModule):
    """This class is used for training and evaluation of GPT-2-based model for
    commit message completion task.

    Args:
        learning_rate: maximum learning rate
        decoder_name_or_path: name or path for pretrained GPT-2 checkpoint
        tokenizer: tokenizer for target sequences (messages)
        num_epochs: total number of epochs (used to calculate total number of steps for LR scheduler)
        num_batches: total number of batches in one epoch (used to calculate total number of steps for LR scheduler)
        num_gpus: total number of GPUs (used to calculate total number of steps for LR scheduler)
    """

    def __init__(
        self,
        learning_rate: float,
        decoder_name_or_path: str,
        tokenizer: GPT2Tokenizer,
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

        # will be logged to W&B
        self.table_data = defaultdict(list)
        self.val_metrics = EvaluationMetrics(do_strings=False, do_tensors=True, prefix="val")
        self.test_metrics = EvaluationMetrics(do_strings=True, do_tensors=False, prefix="test")

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def forward(self, batch):
        self.examples_count += len(batch["msg_input_ids"])
        return self.model(
            input_ids=batch["msg_input_ids"], attention_mask=batch["msg_attention_mask"], labels=batch["msg_labels"]
        )

    def generate_with_prefix(self, batch, **generation_kwargs):
        prefix_fn = PrefixAllowedTokens(
            prefix={i: prefix for i, prefix in enumerate(batch["prefix"])},
            context_len={i: len(msg) for i, msg in enumerate(batch["msg_input_ids"])},
            tokenizer=self._tokenizer,
        )
        return self.model.generate(
            input_ids=batch["msg_input_ids"],
            attention_mask=batch["msg_attention_mask"],
            prefix_allowed_tokens_fn=prefix_fn,
            **generation_kwargs,
        )

    def generate(self, batch, **generation_kwargs):
        return self.model.generate(
            input_ids=batch["msg_input_ids"], attention_mask=batch["msg_attention_mask"], **generation_kwargs
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
            self.log(
                "val_loss_epoch", metrics["val_loss_epoch"], on_step=False, on_epoch=True, prog_bar=True, logger=False
            )
        self.logger.experiment.log(metrics, step=self.examples_count)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # validation on github data: calculating next token prediction metrics
        if dataloader_idx == 0:
            loss, logits = self(batch)[:2]
            self.val_metrics.add_batch(predictions_tensor=logits, references_tensor=batch["msg_labels"])
            return {"loss": loss}
        # validation on marker tests: generating examples
        if dataloader_idx == 1:
            # leave only generated part (without history)
            gen_sequences = self.generate(batch)[:, batch["generation_input_ids"].shape[1] :]

            # remove history from targets
            batch["msg_labels"][batch["msg_labels"] == -100] = self._msg_tokenizer.pad_token_id

            # decode tokenized sequences
            decoded_preds, decoded_trg = self.decode_trg(gen_sequences, batch["msg_labels"])

            # add data to a little table with examples
            self.table_data["Prediction"].extend(decoded_preds)

    def validation_epoch_end(self, outputs):
        # next token prediction metrics
        metrics = self.val_metrics.compute()
        metrics["val_loss"] = torch.stack([x["loss"] for x in outputs[0]]).mean()
        # needed for ModelCheckpoint
        self.log("val_MRR_top5", metrics["val_MRR_top5"], on_step=False, on_epoch=True, prog_bar=True, logger=False)

        # generation examples on marker tests
        table = wandb.Table(dataframe=pd.DataFrame.from_dict(self.table_data))
        self.table_data.clear()
        metrics.update({"marker_tests": table})

        self.logger.experiment.log(metrics, step=self.examples_count)

    def test_step(self, batch, batch_idx):
        # leave only generated part (without history)
        gen_sequences = self.generate(batch)[:, batch["msg_input_ids"].shape[1] :]

        # decode tokenized sequences
        decoded_context, decoded_preds, decoded_trg = self.decode_trg(
            batch["msg_input_ids"], gen_sequences, batch["target"]
        )

        # remove prefix from generated to compute metrics without it
        decoded_preds = [pred[len(prefix) :] for pred, prefix in zip(decoded_preds, batch["prefix"])]
        decoded_trg = [trg[len(prefix) :] for trg, prefix in zip(decoded_trg, batch["prefix"])]

        # add data to metrics
        self.test_metrics.add_batch(predictions=decoded_preds, references=decoded_trg)

        # add data to a little table with examples
        self.table_data["Context"].extend(decoded_context)
        self.table_data["Prefix"].extend(batch["prefix"])
        self.table_data["Prediction"].extend(decoded_preds)
        self.table_data["Target"].extend(decoded_trg)

    def test_epoch_end(self, outputs):
        metrics = self.test_metrics.compute()

        table = wandb.Table(dataframe=pd.DataFrame.from_dict(self.table_data))
        self.table_data.clear()
        metrics.update({"test_examples": table})

        self.logger.experiment.log(metrics, step=self.examples_count)

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
