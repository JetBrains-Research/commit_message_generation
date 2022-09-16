import logging
from collections import defaultdict
from math import log
from typing import Any, Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup

from src.model.configurations import BaseModel, DecoderWrapper, EncoderDecoderWrapper
from src.utils import Batch, BatchTest, BatchTrain, EvaluationMetrics


class CMCModule(pl.LightningModule):
    """This class is used for training and evaluation of various Transformer-based model for
    commit message completion task.

    Args:
        diff_tokenizer: Tokenizer for source sequences (diffs)
        msg_tokenizer: Tokenizer for target sequences (messages)
        model_configuration: Specific model type. Currently supported: `encoder_decoder` and `decoder`.
        preds_artifact_name: An artifact name for saving model predictions as W&B artifact.
        preds_artifact_type: An artifact type for saving model predictions as W&B artifact.
        preds_table_name: A table name for saving model predictions as W&B artifact.
        learning_rate: Learning rate (maximum learning rate if warmup is used).
        weight_decay: Decoupled weight decay to apply in AdamW.
        num_warmup_steps: Number of warmup steps for learning rate scheduler.
        batch_size: Number of examples in a single batch (used to scale LR).
        num_epochs: Total number of epochs (used to calculate total number of steps for LR scheduler)
        num_batches: Total number of batches in one epoch (used to calculate total number of steps for LR scheduler)
        num_gpus: Total number of GPUs (used to calculate total number of steps for LR scheduler)
        generation_kwargs: kwargs for transformers.generation_utils.GenerationMixin.generate
        model_kwargs: All other kwargs will be passed to specific model class.
    """

    def __init__(
        self,
        diff_tokenizer: PreTrainedTokenizerFast,
        msg_tokenizer: PreTrainedTokenizerFast,
        model_configuration: str,
        preds_artifact_name: Optional[str] = None,
        preds_artifact_type: Optional[str] = None,
        preds_table_name: Optional[str] = None,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        num_warmup_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_batches: Optional[int] = None,
        num_epochs: Optional[int] = None,
        num_gpus: Optional[int] = None,
        generation_kwargs: DictConfig = DictConfig({}),
        **model_kwargs,
    ):
        super().__init__()

        self._diff_tokenizer = diff_tokenizer
        self._msg_tokenizer = msg_tokenizer

        self._preds_artifact_name = preds_artifact_name
        self._preds_artifact_type = preds_artifact_type
        self._preds_table_name = preds_table_name

        self.model: BaseModel
        if model_configuration == "encoder_decoder":
            self.model = EncoderDecoderWrapper(
                diff_tokenizer=diff_tokenizer, msg_tokenizer=msg_tokenizer, **model_kwargs
            )
        elif model_configuration == "decoder":
            self.model = DecoderWrapper(tokenizer=msg_tokenizer, **model_kwargs)
        else:
            raise ValueError(
                f'Configuration {model_configuration} is not supported, please use one of "encoder_decoder" or "decoder"'
            )

        self._num_epochs = num_epochs
        self._num_batches = num_batches
        self._batch_size = batch_size
        self._num_gpus = num_gpus

        self.learning_rate = self.adjust_learning_rate(initial_learning_rate=learning_rate)
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps

        self.generation_kwargs = generation_kwargs

        self.save_hyperparameters(logger=False)

        self.num_processed_tokens = 0.0
        self.num_processed_examples = 0.0

        # will be logged to W&B
        self.table_data: Dict[str, List[str]] = defaultdict(list)
        self.val_metrics = EvaluationMetrics(
            do_strings=False,
            do_tensors=True,
            shift=not isinstance(self.model, EncoderDecoderWrapper),
            prefix="val",
        )

    def forward(self, batch: Batch) -> Any:  # type: ignore[override]
        return self.model.forward(batch)

    def generate(self, batch: BatchTest, **kwargs) -> Any:
        if not kwargs:
            kwargs = self.generation_kwargs  # type: ignore[assignment]
        return self.model.generate(batch, **kwargs)

    def training_step(self, batch: BatchTrain, *args, **kwargs):  # type: ignore[override]
        outputs = self(batch)

        self.num_processed_tokens += batch.encoder_attention_mask.sum().item()
        self.num_processed_examples += batch.encoder_attention_mask.shape[0]
        self.log(
            "train_loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=len(batch.encoder_input_ids),
        )
        self.log("num_tokens", self.num_processed_tokens, on_step=True, on_epoch=False, logger=True)
        self.log("num_examples", self.num_processed_examples, on_step=True, on_epoch=False, logger=True)
        return {"loss": outputs.loss}

    def validation_step(self, batch: BatchTrain, *args, **kwargs):  # type: ignore[override]
        outputs = self(batch)
        self.val_metrics.add_batch(predictions_tensor=outputs.logits, references_tensor=batch.labels)
        self.log(
            "val_loss",
            outputs.loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=len(batch.encoder_input_ids),
        )
        return {"loss": outputs.loss}

    def validation_epoch_end(self, outputs):  # type: ignore[override]
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch: BatchTest, *args, **kwargs):  # type: ignore[override]
        gen_sequences = self.generate(batch)
        # leave only generated part (crop context)
        gen_sequences = gen_sequences[:, batch.decoder_input_ids.shape[1] :]

        # decode tokenized sequences
        decoded_source = self.decode_src(batch.encoder_input_ids, skip_special_tokens=True)[0]
        decoded_preds = self.decode_trg(gen_sequences, skip_special_tokens=True)[0]
        decoded_context = self.decode_trg(batch.decoder_input_ids, skip_special_tokens=False)[0]

        # remove prefix from generated sequences to compute metrics without it
        decoded_preds = [pred[len(prefix) :] for pred, prefix in zip(decoded_preds, batch.prefixes)]

        # separate history from corresponding message and get rid of special tokens
        # TODO: looks ugly
        history = []
        for i in range(len(decoded_context)):
            decoded_context[i] = (
                decoded_context[i]
                .replace(self._msg_tokenizer.pad_token, "")  # type: ignore[attr-defined]
                .replace(self._msg_tokenizer.bos_token, "")  # type: ignore[attr-defined]
                .replace(self._msg_tokenizer.eos_token, "")  # type: ignore[attr-defined]
                .replace("[NL]", "\n")
            )
            if self._msg_tokenizer.sep_token in decoded_context[i]:  # type: ignore[attr-defined]
                cur_history = "\n".join(
                    decoded_context[i].split(self._msg_tokenizer.sep_token)[:-1]  # type: ignore[attr-defined]
                )
                history.append(cur_history)
                decoded_context[i] = decoded_context[i].split(self._msg_tokenizer.sep_token)[-1]  # type: ignore[attr-defined]
            else:
                history.append("")

        # add data to a little table with examples
        self.table_data["Diff"].extend(decoded_source)
        self.table_data["History"].extend(history)
        self.table_data["Context"].extend(decoded_context)
        self.table_data["Prefix"].extend(batch.prefixes)
        self.table_data["Prediction"].extend(decoded_preds)
        self.table_data["Target"].extend([target.replace("[NL]", "\n") for target in batch.targets])

    def test_epoch_end(self, *args, **kwargs):
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.log_table("test_examples", dataframe=pd.DataFrame.from_dict(self.table_data))

            # upload predictions to wandb as artifact
            if self._preds_artifact_name and self._preds_artifact_type and self._preds_table_name:
                artifact = wandb.Artifact(
                    self._preds_artifact_name,
                    type=self._preds_artifact_type,
                    metadata={"tags": self.logger.experiment.tags if self.logger.experiment.tags else None},
                )
                artifact.add(wandb.Table(dataframe=pd.DataFrame.from_dict(self.table_data)), self._preds_table_name)
                self.logger.experiment.log_artifact(artifact, aliases=self._preds_table_name)

    def configure_optimizers(self):
        if not self.learning_rate:
            logging.warning("Learning rate is not set, proceeding with default value 1e-3")
            self.learning_rate = 1e-3

        if not self.weight_decay:
            logging.warning("Weight decay is not set, proceeding with default value 1e-2")
            self.weight_decay = 1e-2

        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        if not (self._num_epochs and self._batch_size and self._num_batches and self._num_gpus):
            logging.warning("Number of batches is not set, proceeding without warmup scheduler")
            return optimizer

        if not self.num_warmup_steps:
            logging.warning("Number of warmup steps is not set, proceeding without warmup scheduler")
            return optimizer

        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.num_warmup_steps // self._num_gpus,
                num_training_steps=self._num_epochs * self._num_batches,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def decode_src(self, *args, **kwargs):
        return tuple(self._diff_tokenizer.batch_decode(arg, **kwargs) for arg in args)

    def decode_trg(self, *args, **kwargs):
        return tuple(self._msg_tokenizer.batch_decode(arg, **kwargs) for arg in args)

    def adjust_learning_rate(
        self, initial_batch_size: Optional[int] = None, initial_learning_rate: Optional[float] = None
    ) -> float:
        assert self._batch_size
        if not initial_learning_rate:
            # take formula from `Scaling Laws for Neural Language Models` paper
            # and scale linearly with batch size (it was 512 in the paper)
            initial_batch_size = 512
            initial_learning_rate = 0.003239 - 0.0001395 * log(self.model.num_parameters(exclude_embeddings=True))
        initial_learning_rate = initial_learning_rate * self._batch_size
        if initial_batch_size:
            return initial_learning_rate / initial_batch_size
        return initial_learning_rate
