import logging
from collections import defaultdict
from copy import copy
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    GPT2LMHeadModel,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    RobertaForCausalLM,
    RobertaModel,
    get_linear_schedule_with_warmup,
)

from src.utils import EvaluationMetrics, PrefixAllowedTokens


class EncoderDecoderModule(pl.LightningModule):
    """This class is used for training and evaluation of seq2seq Transformer model for
    commit message completion task.

    It is possible to either use pretrained models or initialize from scratch.

    Args:
        diff_tokenizer: tokenizer for source sequences (diffs)
        msg_tokenizer: tokenizer for target sequences (messages)
        wandb_artifact_name: an artifact name for saving model predictions as W&B artifact
        wandb_artifact_type: an artifact type for saving model predictions as W&B artifact
        wandb_table_name: a table name for saving model predictions as W&B artifact
        learning_rate: maximum learning rate
        num_epochs: total number of epochs (used to calculate total number of steps for LR scheduler)
        num_batches: total number of batches in one epoch (used to calculate total number of steps for LR scheduler)
        num_gpus: total number of GPUs (used to calculate total number of steps for LR scheduler)
        num_layers_encoder: if `encoder_name_or_path` is None, encoder will be initialized
            from scratch with given number of layers; else, if given number of layers is less than number of layers in
            pretrained checkpoint, it will be uniformly picked
        num_layers_decoder: If `decoder_name_or_path` is None, decoder will be initialized
            from scratch with given number of layers; else, if given number of layers is less than number of layers in
            pretrained checkpoint, it will be uniformly picked
        encoder_name_or_path: Optional – name or path to pretrained checkpoint to initialize encoder with
        decoder_name_or_path: Optional – name or path to pretrained checkpoint to initialize decoder with
        encoder_model_type: Optional – if encoder is initialized from scratch, this specific model class will be used
        decoder_model_type: Optional – if decoder is initialized from scratch, this specific model class will be used
        tie_encoder_decoder: If set to `True`, encoder and decoder will share the same parameters
        tie_word_embeddings: If set to `True`, encoder and decoder will share the same parameters for embedding layers
        generation_kwargs: kwargs for transformers.generation_utils.GenerationMixin.generate
    """

    def __init__(
        self,
        diff_tokenizer: PreTrainedTokenizerFast,
        msg_tokenizer: PreTrainedTokenizerFast,
        wandb_artifact_name: Optional[str] = None,
        wandb_artifact_type: Optional[str] = None,
        wandb_table_name: Optional[str] = None,
        learning_rate: Optional[float] = None,
        num_epochs: Optional[int] = None,
        num_batches: Optional[int] = None,
        num_gpus: Optional[int] = None,
        num_layers_encoder: Optional[int] = None,
        num_layers_decoder: Optional[int] = None,
        encoder_name_or_path: Optional[str] = None,
        decoder_name_or_path: Optional[str] = None,
        encoder_model_type: Optional[str] = None,
        decoder_model_type: Optional[str] = None,
        tie_encoder_decoder: Optional[bool] = None,
        tie_word_embeddings: Optional[bool] = None,
        generation_kwargs: Optional[DictConfig] = None,
        **kwargs,
    ):
        super().__init__()

        self._diff_tokenizer = diff_tokenizer
        self._msg_tokenizer = msg_tokenizer

        self._wandb_artifact_name = wandb_artifact_name
        self._wandb_artifact_type = wandb_artifact_type
        self._wandb_table_name = wandb_table_name

        self._num_epochs = num_epochs
        self._num_batches = num_batches
        self._num_gpus = num_gpus
        self.learning_rate = learning_rate

        self.generation_kwargs = generation_kwargs

        self.save_hyperparameters()

        encoder = self.prepare_model(
            encoder_or_decoder="encoder",
            model_type=encoder_model_type,
            name_or_path=encoder_name_or_path,
            num_layers=num_layers_encoder,
        )
        decoder = self.prepare_model(
            encoder_or_decoder="decoder",
            model_type=decoder_model_type,
            name_or_path=decoder_name_or_path,
            num_layers=num_layers_decoder,
        )

        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=encoder.config, decoder_config=decoder.config
        )
        config.encoder.tie_encoder_decoder = tie_encoder_decoder
        config.decoder.tie_encoder_decoder = tie_encoder_decoder
        config.tie_encoder_decoder = tie_encoder_decoder
        config.tie_word_embeddings = tie_word_embeddings

        self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder, config=config)

        # cache is currently not supported by EncoderDecoder framework
        self.model.decoder.config.use_cache = False

        # will be logged to W&B
        self.table_data = defaultdict(list)
        self.val_metrics = EvaluationMetrics(do_strings=False, do_tensors=True, prefix="val")

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def prepare_model(
        self,
        encoder_or_decoder: str,
        model_type: str,
        num_layers: Optional[int] = None,
        name_or_path: Optional[str] = None,
    ) -> PreTrainedModel:
        """
        Initializes either encoder or decoder for further use in EncoderDecoderModel class.

        Args:
            encoder_or_decoder: Pass `encoder` to correctly initialize any model as seq2seq encoder.
              Pass `decoder` to correctly initialize any model as seq2seq decoder.
            model_type: Necessary for training from scratch. Corresponding model class will be used.
              Currently supported: `bert`, `roberta`, `gpt2`.
            num_layers: Optional – number of layers. If pretrained model is used and `num_layers` is less than
              actual number of layers in the model, `num_layers` layers will be picked uniformly. When empty,
              default number of layers will be used.
            name_or_path: Optional – name on HuggingFace Hub or path to pretrained model weights.

        Returns:
            initialized model for further use in EncoderDecoderModel class
        """
        # use pretrained model
        if name_or_path:
            if encoder_or_decoder == "encoder":
                model = AutoModel.from_pretrained(name_or_path)
            else:
                model = AutoModelForCausalLM.from_pretrained(name_or_path, is_decoder=True, add_cross_attention=True)

            # remove layers if necessary
            if num_layers is not None:
                if model.config.model_type in ["bert", "roberta", "gpt2"]:
                    if (
                        model.config.model_type in ["bert", "roberta"]
                        and num_layers < model.config.num_hidden_layers
                        or model.config.model_type == "gpt2"
                        and num_layers < model.config.n_layer
                    ):
                        model = EncoderDecoderModule.remove_layers_from_model(model, num_layers)
                else:
                    logging.warning("Unknown model type, default number of layers is used")
        elif num_layers:
            # use randomly initialized model
            config = AutoConfig.for_model(model_type=model_type)

            # set specified number of hidden layers
            if config.model_type == "gpt2":
                config.n_layer = num_layers
            elif config.model_type in ["bert", "roberta"]:
                config.num_hidden_layers = num_layers
            else:
                logging.warning("Unknown model type, default number of layers is used")

            # update vocabulary size according to corresponding tokenizer
            if encoder_or_decoder == "encoder":
                config.vocab_size = len(self._diff_tokenizer)
                model = AutoModel.from_config(config=config)
            else:
                config.vocab_size = len(self._msg_tokenizer)
                config.is_decoder = True
                config.add_cross_attention = True
                model = AutoModelForCausalLM.from_config(config=config)
        else:
            raise ValueError(
                f"Unable to initialize {encoder_or_decoder}. You have to specify either `num_layers` and `model_type` to train from scratch or `name_or_path` to load pretrained model"
            )

        return model

    def forward(self, batch):
        return self.model(
            input_ids=batch["diff_input_ids"],
            attention_mask=batch["diff_attention_mask"],
            decoder_input_ids=batch["msg_input_ids"],
            decoder_attention_mask=batch["msg_attention_mask"],
            labels=batch["msg_labels"],
        )

    def generate(self, batch, **generation_kwargs):
        prefix_fn = PrefixAllowedTokens(
            prefix={i: prefix for i, prefix in enumerate(batch["msg_prefix"])},
            context_len={i: len(msg) for i, msg in enumerate(batch["msg_input_ids"])},
            tokenizer=self._msg_tokenizer,
        )

        if "min_length" in generation_kwargs:
            generation_kwargs["min_length"] += batch["msg_input_ids"].shape[1]

        if "max_length" in generation_kwargs:
            generation_kwargs["max_length"] += batch["msg_input_ids"].shape[1]

        return self.model.generate(
            input_ids=batch["diff_input_ids"],
            attention_mask=batch["diff_attention_mask"],
            decoder_input_ids=batch["msg_input_ids"],
            decoder_attention_mask=batch["msg_attention_mask"],
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
        # validation on github data: calculating next token prediction metrics
        if not dataloader_idx:
            loss, logits = self(batch)[:2]
            self.val_metrics.add_batch(predictions_tensor=logits, references_tensor=batch["msg_labels"])
            return {"loss": loss}
        # validation on marker tests: generating examples
        if dataloader_idx == 1:
            # leave only generated part (without history)
            gen_sequences = self.generate(
                batch,
                pad_token_id=self._msg_tokenizer.eos_token_id,
                eos_token_id=198,
                early_stopping=True,
                no_repeat_ngram_size=4,
                num_beams=5,
                min_length=5,
                max_length=10,
            )[:, batch["msg_input_ids"].shape[1] :]

            # decode tokenized sequences
            decoded_source = self.decode_src(batch["diff_input_ids"])[0]
            decoded_preds = self.decode_trg(gen_sequences)[0]

            # add data to a little table with examples
            self.table_data["Diff"].extend(decoded_source)
            self.table_data["Prediction"].extend(decoded_preds)

    def validation_epoch_end(self, outputs):
        # next token prediction metrics
        metrics = self.val_metrics.compute()

        # for case with two dataloaders: usual validation & marker test
        if len(outputs) == 2 and not outputs[1]:
            outputs = outputs[0]

        metrics["val_loss"] = torch.stack([x["loss"] for x in outputs]).mean()

        # needed for ModelCheckpoint
        self.log("val_MRR_top5", metrics["val_MRR_top5"], on_step=False, on_epoch=True, prog_bar=True, logger=False)

        if self.table_data:
            # generation examples on marker tests
            table = wandb.Table(dataframe=pd.DataFrame.from_dict(self.table_data))
            self.table_data.clear()
            metrics.update({"marker_tests": table})

        self.logger.experiment.log(metrics, step=self.examples_count)

    def test_step(self, batch, batch_idx):
        # leave only generated part (without history)
        gen_sequences = self.generate(
            batch,
            early_stopping=True,
            eos_token_id=198,
            pad_token_id=self._msg_tokenizer.eos_token_id,
            **self.generation_kwargs,
        )[:, batch["msg_input_ids"].shape[1] :]

        # decode tokenized sequences
        decoded_source = self.decode_src(batch["diff_input_ids"])[0]
        decoded_context, decoded_preds = self.decode_trg(batch["msg_input_ids"], gen_sequences)

        # remove prefix from predicted to compute metrics without it
        decoded_preds = [pred[len(prefix) :] for pred, prefix in zip(decoded_preds, batch["msg_prefix"])]

        history = []
        for i in range(len(decoded_context)):
            if "\n" in decoded_context[i]:
                history.append("\n".join(decoded_context[i].split("\n")[:-1]))
                decoded_context[i] = decoded_context[i].split("\n")[-1]
            else:
                history.append("")

        # add data to a little table with examples
        self.table_data["Diff"].extend(decoded_source)
        self.table_data["History"].extend(history)
        self.table_data["Context"].extend(decoded_context)
        self.table_data["Prefix"].extend(batch["msg_prefix"])
        self.table_data["Prediction"].extend(decoded_preds)
        self.table_data["Target"].extend(batch["msg_target"])

    def test_epoch_end(self, outputs):
        table = wandb.Table(dataframe=pd.DataFrame.from_dict(self.table_data))
        self.table_data.clear()
        self.logger.experiment.log({"test_examples": table}, step=self.examples_count)

        if self._wandb_artifact_name and self._wandb_artifact_type and self._wandb_table_name:
            artifact = wandb.Artifact(self._wandb_artifact_name, type=self._wandb_artifact_type)
            artifact.add(table, self._wandb_table_name)
            self.logger.experiment.log_artifact(artifact)

    def configure_optimizers(self):
        if not self.learning_rate:
            logging.warning("Learning rate is not set, proceeding with default value 1e-3")
            self.learning_rate = 1e-3

        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        if not (self._num_epochs and self._num_batches and self._num_gpus):
            logging.warning("Number of batches is not set, proceeding without warmup scheduler")
            return optimizer

        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer, 4000 // self._num_gpus, self._num_epochs * self._num_batches
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def decode_src(self, *args):
        return tuple(self._diff_tokenizer.batch_decode(arg, skip_special_tokens=True) for arg in args)

    def decode_trg(self, *args):
        return tuple(self._msg_tokenizer.batch_decode(arg, skip_special_tokens=True) for arg in args)

    @staticmethod
    def remove_layers_from_model(teacher: PreTrainedModel, num_layers: int) -> PreTrainedModel:
        if isinstance(teacher, RobertaForCausalLM):
            student_config = copy(teacher.config)
            student_config.num_hidden_layers = num_layers
            student = RobertaForCausalLM(config=student_config)

            # copy all embeddings
            student.roberta.embeddings.word_embeddings = teacher.roberta.embeddings.word_embeddings
            student.roberta.embeddings.position_embeddings = teacher.roberta.embeddings.position_embeddings
            student.roberta.embeddings.token_type_embeddings = teacher.roberta.embeddings.token_type_embeddings
            student.roberta.embeddings.LayerNorm = teacher.roberta.embeddings.LayerNorm
            student.roberta.embeddings.dropout = teacher.roberta.embeddings.dropout

            # uniformly pick from middle layers from teacher
            # it is basically np.linspace(0, teacher_config.num_hidden_layers,
            #                             num=student_config.num_hidden_layers, endpoint=True)
            step = (teacher.config.num_hidden_layers - 1) / (student_config.num_hidden_layers - 1)
            for student_layer, teacher_layer in enumerate(
                int(i * step) for i in range(student_config.num_hidden_layers)
            ):
                student.roberta.encoder.layer[student_layer] = teacher.roberta.encoder.layer[teacher_layer]
        elif isinstance(teacher, RobertaModel):
            student_config = copy(teacher.config)
            student_config.num_hidden_layers = num_layers
            student = RobertaModel(config=student_config)

            # copy all embeddings
            student.embeddings.word_embeddings = teacher.embeddings.word_embeddings
            student.embeddings.position_embeddings = teacher.embeddings.position_embeddings
            student.embeddings.token_type_embeddings = teacher.embeddings.token_type_embeddings
            student.embeddings.LayerNorm = teacher.embeddings.LayerNorm
            student.embeddings.dropout = teacher.embeddings.dropout

            # uniformly pick from middle layers from teacher
            # it is basically np.linspace(0, teacher_config.num_hidden_layers,
            #                             num=student_config.num_hidden_layers, endpoint=True)
            step = (teacher.config.num_hidden_layers - 1) / (student_config.num_hidden_layers - 1)
            for student_layer, teacher_layer in enumerate(
                int(i * step) for i in range(student_config.num_hidden_layers)
            ):
                student.encoder.layer[student_layer] = teacher.encoder.layer[teacher_layer]
        elif isinstance(teacher, GPT2LMHeadModel):
            student_config = copy(teacher.config)
            student_config.n_layer = num_layers

            student = GPT2LMHeadModel(config=student_config)

            # Copying all embeddings
            student.transformer.wte = teacher.transformer.wte
            student.transformer.wpe = teacher.transformer.wpe
            student.transformer.drop = teacher.transformer.drop
            # Maybe there is something else in BERT that need to be copied!
            # Specific thing for GPT2LMHead. Not necessary for BERT
            student.tie_weights()
            # Uniformly pick from middle layers from teacher
            # It is basically np.linspace(0, teacher_config.n_layer, num=student_config.n_layer, endpoint=True)
            step = (teacher.config.n_layer - 1) / (student_config.n_layer - 1)
            for student_layer, teacher_layer in enumerate(int(i * step) for i in range(student_config.n_layer)):
                student.transformer.h[student_layer] = teacher.transformer.h[teacher_layer]
        return student
