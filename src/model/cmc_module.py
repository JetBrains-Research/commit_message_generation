import logging
from math import log
from typing import Any, Dict, List, Optional

import jsonlines
import pytorch_lightning as pl
import torch
import wandb
from torch.nn import LayerNorm
from torch.optim import AdamW
from transformers import PreTrainedTokenizerFast, get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names

from conf import (
    BaseDecoderConfig,
    BaseEncoderDecoderConfig,
    BaseModelConfig,
    BaseRACEConfig,
    BaseSeq2SeqConfig,
)
from src.model.configurations import (
    BaseModel,
    DecoderWrapper,
    EncoderDecoderWrapper,
    RACEWrapper,
    Seq2SeqWrapper,
)
from src.utils import (
    Batch,
    BatchTest,
    BatchTrain,
    EvaluationMetrics,
    PrefixAllowedTokens,
    VocabPrefixTree,
)


class CMCModule(pl.LightningModule):
    """This class is used for training and evaluation of various Transformer-based models for
    a commit message completion task.

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
        generation_kwargs: kwargs for transformers.generation_utils.GenerationMixin.generate
        model_kwargs: All other kwargs will be passed to specific model class.
    """

    def __init__(
        self,
        model_cfg: BaseModelConfig,
        diff_tokenizer: PreTrainedTokenizerFast,
        msg_tokenizer: PreTrainedTokenizerFast,
        preds_artifact_name: Optional[str] = None,
        preds_artifact_type: Optional[str] = None,
        preds_table_name: Optional[str] = None,
        learning_rate: Optional[float] = None,
        initial_batch_size: Optional[int] = None,
        weight_decay: Optional[float] = None,
        num_warmup_steps: Optional[int] = None,
        ratio_warmup_steps: Optional[float] = None,
        batch_size: Optional[int] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()

        self._diff_tokenizer = diff_tokenizer
        self._msg_tokenizer = msg_tokenizer

        self._vocab_trie = None

        self.model: BaseModel = self._init_model(model_cfg)

        self.learning_rate = self.adjust_learning_rate(
            batch_size=batch_size, initial_learning_rate=learning_rate, initial_batch_size=initial_batch_size
        )
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps
        self.ratio_warmup_steps = ratio_warmup_steps

        self._preds_artifact_name = preds_artifact_name
        self._preds_artifact_type = preds_artifact_type
        self._preds_table_name = preds_table_name

        self.generation_kwargs = generation_kwargs

        self.save_hyperparameters(logger=False)

        self.num_processed_tokens = 0.0
        self.num_processed_examples = 0.0

        # will be logged to W&B
        self.preds: List[Dict[str, str]] = []
        self.train_metrics = EvaluationMetrics(
            do_strings=False,
            do_tensors=True,
            shift=isinstance(self.model, DecoderWrapper),
            prefix="train",
        )
        self.val_metrics = EvaluationMetrics(
            do_strings=False,
            do_tensors=True,
            shift=isinstance(self.model, DecoderWrapper),
            prefix="val",
        )

    def _init_model(self, model_cfg: BaseModelConfig) -> BaseModel:
        """Initializes a correct model type based on passed parameters.

        Args:
            model_cfg: Config with model parameters.

        Returns:
            Initialized model.
        """
        if model_cfg.configuration == "encoder_decoder":
            model_cfg = BaseEncoderDecoderConfig(**model_cfg)  # type: ignore[arg-type]
            return EncoderDecoderWrapper(
                diff_tokenizer=self._diff_tokenizer,
                msg_tokenizer=self._msg_tokenizer,
                encoder_context_max_len=model_cfg.encoder_context_max_len,
                decoder_context_max_len=model_cfg.decoder_context_max_len,
                encoder_name_or_path=model_cfg.encoder_name_or_path,
                decoder_name_or_path=model_cfg.decoder_name_or_path,
                num_layers_encoder=model_cfg.num_layers_encoder,
                num_layers_decoder=model_cfg.num_layers_decoder,
                encoder_model_type=model_cfg.encoder_model_type,
                decoder_model_type=model_cfg.decoder_model_type,
                tie_encoder_decoder=model_cfg.tie_encoder_decoder,
                tie_word_embeddings=model_cfg.tie_word_embeddings,
            )
        elif model_cfg.configuration == "decoder":
            model_cfg = BaseDecoderConfig(**model_cfg)  # type: ignore[arg-type]
            return DecoderWrapper(tokenizer=self._msg_tokenizer, decoder_name_or_path=model_cfg.decoder_name_or_path)
        elif model_cfg.configuration == "seq2seq":
            model_cfg = BaseSeq2SeqConfig(**model_cfg)  # type: ignore[arg-type]
            return Seq2SeqWrapper(tokenizer=self._msg_tokenizer, name_or_path=model_cfg.name_or_path)
        elif model_cfg.configuration == "race":
            model_cfg = BaseRACEConfig(**model_cfg)  # type: ignore[arg-type]
            return RACEWrapper(tokenizer=self._msg_tokenizer, name_or_path=model_cfg.name_or_path)

        raise ValueError(f"Current configuration ({model_cfg.configuration}) is not supported")

    @property
    def vocab_trie(self):
        if self._vocab_trie is None:
            self._vocab_trie = VocabPrefixTree(self._msg_tokenizer)
        return self._vocab_trie

    def forward(self, batch: Batch) -> Any:  # type: ignore[override]
        return self.model.forward(batch)

    def generate(self, batch: BatchTest, **kwargs) -> Any:
        if not kwargs:
            kwargs = self.generation_kwargs  # type: ignore[assignment]

        prefix_fn = None
        if any(prefix for prefix in batch.prefixes):
            prefix_fn = PrefixAllowedTokens(
                prefix={i: prefix for i, prefix in enumerate(batch.prefixes)},
                context_len={i: len(msg) for i, msg in enumerate(batch.decoder_input_ids)},
                trie=self.vocab_trie,
                tokenizer=self._msg_tokenizer,
            )

        return self.model.generate(
            batch,
            **kwargs,
            prefix_allowed_tokens_fn=prefix_fn,
            pad_token_id=self._msg_tokenizer.pad_token_id,  # type: ignore[attr-defined]
            bos_token_id=self._msg_tokenizer.bos_token_id,  # type: ignore[attr-defined]
            eos_token_id=self._msg_tokenizer.eos_token_id,  # type: ignore[attr-defined]
        )

    def training_step(self, batch: BatchTrain, *args, **kwargs):  # type: ignore[override]
        outputs = self(batch)

        with torch.no_grad():
            cur_metrics = self.train_metrics.add_batch(
                predictions_tensor=outputs.logits, references_tensor=batch.labels
            )
        self.num_processed_tokens += batch.encoder_attention_mask.sum().item()
        self.num_processed_examples += batch.encoder_attention_mask.shape[0]

        self.log_dict(cur_metrics, on_step=True, on_epoch=False, logger=True)
        self.log(
            "train_loss",
            outputs.loss,
            on_step=True,
            on_epoch=True,
            logger=True,
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
            batch_size=len(batch.encoder_input_ids),
            sync_dist=True,
        )
        return {"loss": outputs.loss}

    def validation_epoch_end(self, outputs):  # type: ignore[override]
        self.log_dict(self.val_metrics.compute(), on_step=False, on_epoch=True, logger=True)

    def _postprocess_generated(self, batch: BatchTest, predictions: torch.Tensor) -> List[Dict[str, str]]:
        """Decodes predictions and context.

        Args:
            batch: Model inputs.
            predictions: Model predictions.

        Returns:
            A dict with decoded sources/predictions.
        """
        decoded_preds = self.decode_trg(predictions, skip_special_tokens=True)[0]
        decoded_context = self.decode_trg(batch.decoder_input_ids, skip_special_tokens=True)[0]

        return [
            {"Context": context, "Prefix": prefix, "Prediction": pred[len(prefix) :], "Target": target}
            for context, prefix, pred, target in zip(decoded_context, batch.prefixes, decoded_preds, batch.targets)
        ]

    def write_preds(self, preds: List[Dict[str, str]]) -> None:
        if len(self.preds) > 100:
            with jsonlines.open(f"{self._preds_table_name}.jsonl", "a") as writer:
                writer.write_all(self.preds)
            self.preds = []

        self.preds.extend(preds)

    def test_step(self, batch: BatchTest, *args, **kwargs):  # type: ignore[override]
        predictions = self.generate(batch)
        # leave only generated part (crop context)
        predictions = predictions[:, batch.decoder_input_ids.shape[1] :]

        # decode & postprocess data
        string_results = self._postprocess_generated(batch, predictions)

        # write to file
        self.write_preds(string_results)

    def test_epoch_end(self, *args, **kwargs):
        if self.preds:
            with jsonlines.open(f"{self._preds_table_name}.jsonl", "a") as writer:
                writer.write_all(self.preds)

        # is W&B is used, upload predictions as artifact
        if isinstance(self.logger, pl.loggers.WandbLogger):
            artifact = wandb.Artifact(
                self._preds_artifact_name,
                type=self._preds_artifact_type,
                metadata={"tags": self.logger.experiment.tags if self.logger.experiment.tags else None},
                incremental=True,
            )
            artifact.add_file(f"{self._preds_table_name}.jsonl")
            self.logger.experiment.log_artifact(artifact)

    def configure_optimizers(self):
        if self.learning_rate is None:
            logging.warning("Learning rate is not set, proceeding with default value 1e-3")
            self.learning_rate = 1e-3

        if self.weight_decay is None:
            logging.warning("Weight decay is not set, proceeding with default value 1e-2")
            self.weight_decay = 1e-2

        # reusing implementation from Huggingface Transformers to skip LayerNorm in weight decay
        # https://github.com/huggingface/transformers/blob/v4.26.1/src/transformers/trainer.py#L1019
        decay_parameters = get_parameter_names(self.model, [LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        if self.num_warmup_steps is None and self.ratio_warmup_steps is None:
            logging.warning("Number of warmup steps is not set, proceeding without warmup scheduler")
            return optimizer

        if self.ratio_warmup_steps is not None:
            if self.num_warmup_steps is not None:
                logging.warning(
                    "Both `num_warmup_steps` and `ratio_warmup_steps` are defined. Will use `ratio_warmup_steps`."
                )
            num_warmup_steps = self.ratio_warmup_steps * self.trainer.estimated_stepping_batches
            logging.info(
                f"Warmup: {self.ratio_warmup_steps * 100:.2}% of total training steps ({self.trainer.estimated_stepping_batches})= {num_warmup_steps} steps"
            )
        else:
            num_warmup_steps = self.num_warmup_steps

        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps // self.trainer.num_devices,
                num_training_steps=self.trainer.estimated_stepping_batches // self.trainer.num_devices,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def save_pretrained(self, path: str) -> None:
        self.model.save_pretrained(path)

    def decode_src(self, *args, **kwargs):
        return tuple(self._diff_tokenizer.batch_decode(arg, **kwargs) for arg in args)

    def decode_trg(self, *args, **kwargs):
        return tuple(self._msg_tokenizer.batch_decode(arg, **kwargs) for arg in args)

    def adjust_learning_rate(
        self,
        batch_size: Optional[int],
        initial_batch_size: Optional[int] = None,
        initial_learning_rate: Optional[float] = None,
    ) -> float:

        # when LR is not passed explicitly, take the formula from `Scaling Laws for Neural Language Models`
        # and scale linearly with batch size (it was 512 in the paper)
        if not initial_learning_rate:
            initial_batch_size = 512
            initial_learning_rate = 0.003239 - 0.0001395 * log(self.model.num_parameters(exclude_embeddings=True))

        # when given initial batch size, scale LR linearly
        if initial_batch_size and batch_size:
            return initial_learning_rate * batch_size / initial_batch_size

        return initial_learning_rate
