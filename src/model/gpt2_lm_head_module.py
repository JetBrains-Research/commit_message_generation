from collections import defaultdict
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from transformers import (
    AdamW,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)

from src.utils import EvaluationMetrics, PrefixAllowedTokens


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
        decoder_name_or_path: str,
        tokenizer: GPT2Tokenizer,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        num_batches: Optional[int] = None,
        num_gpus: Optional[int] = None,
        generation_kwargs: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()
        self._tokenizer = tokenizer

        self._num_epochs = num_epochs
        self._num_batches = num_batches
        self._num_gpus = num_gpus
        self.learning_rate = learning_rate

        self.generation_kwargs = generation_kwargs
        self.save_hyperparameters()

        # use pretrained GPT-2 as decoder
        self.model = GPT2LMHeadModel.from_pretrained(decoder_name_or_path)

        # will be logged to W&B
        self.table_data = defaultdict(list)
        self.val_metrics = EvaluationMetrics(do_strings=False, do_tensors=True, prefix="val")
        self.test_metrics = EvaluationMetrics(do_strings=True, do_tensors=False, prefix="test")

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def forward(self, batch):
        return self.model(
            input_ids=batch["msg_input_ids"], attention_mask=batch["msg_attention_mask"], labels=batch["msg_labels"]
        )

    def generate(self, batch, **generation_kwargs):
        prefix_fn = PrefixAllowedTokens(
            prefix={i: prefix for i, prefix in enumerate(batch["msg_prefix"])},
            context_len={i: len(msg) for i, msg in enumerate(batch["msg_input_ids"])},
            tokenizer=self._tokenizer,
        )

        if "min_length" in generation_kwargs:
            generation_kwargs["min_length"] += batch["msg_input_ids"].shape[1]

        if "max_length" in generation_kwargs:
            generation_kwargs["max_length"] += batch["msg_input_ids"].shape[1]

        return self.model.generate(
            input_ids=batch["msg_input_ids"],
            attention_mask=batch["msg_attention_mask"],
            prefix_allowed_tokens_fn=prefix_fn,
            **generation_kwargs,
        )

    def training_step(self, batch, batch_idx):
        self.examples_count += len(batch["diff_input_ids"])
        loss, logits = self(batch)[:2]
        self.logger.experiment.log({"train_loss_step": loss}, step=self.examples_count)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.log({"train_loss_epoch": train_loss_mean}, step=self.examples_count)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        loss, logits = self(batch)[:2]
        self.val_metrics.add_batch(predictions_tensor=logits, references_tensor=batch["msg_labels"])
        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        # next token prediction metrics
        metrics = self.val_metrics.compute()
        metrics["val_loss"] = torch.stack([x["loss"] for x in outputs]).mean()
        # needed for ModelCheckpoint
        self.log("val_MRR_top5", metrics["val_MRR_top5"], on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.logger.experiment.log(metrics, step=self.examples_count)

    def test_step(self, batch, batch_idx):
        # leave only generated part (without history)
        gen_sequences = self.generate(
            batch,
            early_stopping=True,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=198,
            **self.generation_kwargs,
        )[:, batch["msg_input_ids"].shape[1] :]

        # decode tokenized sequences
        decoded_context, decoded_preds = self.decode_trg(batch["msg_input_ids"], gen_sequences)

        # remove prefix from predicted to compute metrics without it
        decoded_preds = [pred[len(prefix) :] for pred, prefix in zip(decoded_preds, batch["msg_prefix"])]

        # add data to metrics
        self.test_metrics.add_batch(predictions=decoded_preds, references=batch["msg_target"])

        history = []
        for i in range(len(decoded_context)):
            if "\n" in decoded_context[i]:
                history.append("\n".join(decoded_context[i].split("\n")[:-1]))
                decoded_context[i] = decoded_context[i].split("\n")[-1]
            else:
                history.append("")

        # add data to a little table with examples
        self.table_data["History"].extend(history)
        self.table_data["Context"].extend(decoded_context)
        self.table_data["Prefix"].extend(batch["msg_prefix"])
        self.table_data["Prediction"].extend(decoded_preds)
        self.table_data["Target"].extend(batch["msg_target"])

    def test_epoch_end(self, outputs):
        metrics = self.test_metrics.compute()

        table = wandb.Table(dataframe=pd.DataFrame.from_dict(self.table_data))
        self.table_data.clear()
        metrics.update({"test_examples": table})

        self.logger.experiment.log(metrics, step=self.examples_count)

    def decode_trg(self, *args):
        return tuple(self._tokenizer.batch_decode(arg, skip_special_tokens=True) for arg in args)

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
