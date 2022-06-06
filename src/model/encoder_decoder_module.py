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
    EncoderDecoderModel,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaModel,
    get_linear_schedule_with_warmup,
)

from src.utils import EvaluationMetrics, PrefixAllowedTokens


class EncoderDecoderModule(pl.LightningModule):
    """This class is used for training and evaluation of Transformer model for
    commit message completion task.

    More specifically, RoBERTa is used as an encoder and GPT-2 is used as a decoder.
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
        num_layers_decoder: if `decoder_name_or_path` is None, decoder will be initialized
            from scratch with given number of layers; else, if given number of layers is less than number of layers in
            pretrained checkpoint, it will be uniformly picked
        encoder_name_or_path: use to initialize encoder with pretrained checkpoint
        decoder_name_or_path: use to initialize decoder with pretrained checkpoint
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

        if encoder_name_or_path:
            # use pretrained RoBERTa as encoder
            encoder = RobertaModel.from_pretrained(encoder_name_or_path)
            # resize embeddings to match vocabulary size
            encoder.resize_token_embeddings(len(self._diff_tokenizer))
            # remove layers if necessary
            if num_layers_encoder is not None and num_layers_encoder < encoder.config.num_hidden_layers:
                encoder = EncoderDecoderModule.remove_layers_from_model(encoder, num_layers_encoder, is_gpt=False)
        elif num_layers_encoder:
            # use randomly initialized RoBERTa as encoder
            encoder_config = RobertaConfig()
            encoder_config.vocab_size = self._diff_tokenizer.vocab_size
            encoder_config.num_hidden_layers = num_layers_encoder
            encoder = RobertaModel(config=encoder_config)
            # resize embeddings to match vocabulary size
            encoder.resize_token_embeddings(len(self._diff_tokenizer))
        else:
            raise ValueError(
                "You have to specify either `encoder_num_layers` for training from scratch \
                                          or `encoder_name_or_path` for loading pretrained model"
            )

        if decoder_name_or_path:
            # use pretrained GPT-2 as decoder
            config = GPT2Config.from_pretrained(decoder_name_or_path)
            config.is_decoder = True
            config.add_cross_attention = True
            decoder = GPT2LMHeadModel.from_pretrained(decoder_name_or_path, config=config)
            # remove layers if necessary
            if num_layers_decoder is not None and num_layers_decoder < decoder.config.n_layer:
                decoder = EncoderDecoderModule.remove_layers_from_model(decoder, num_layers_decoder, is_gpt=True)
        elif num_layers_decoder:
            # use randomly initialized GPT-2 as decoder
            decoder_config = GPT2Config()
            decoder_config.n_layer = num_layers_decoder
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            decoder = GPT2LMHeadModel(config=decoder_config)
        else:
            raise ValueError(
                "You have to specify either `decoder_num_layers` for training from scratch \
                                          or `decoder_name_or_path` for loading pretrained model"
            )

        self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

        # cache is currently not supported by EncoderDecoder framework
        self.model.decoder.config.use_cache = False

        # do not tie output embeddings to input embeddings
        self.model.config.tie_word_embeddings = False

        # will be logged to W&B
        self.table_data = defaultdict(list)
        self.val_metrics = EvaluationMetrics(do_strings=False, do_tensors=True, prefix="val")

        # to make logs for different batch sizes prettier
        self.examples_count = 0

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
    def remove_layers_from_model(teacher, num_layers, is_gpt):
        if not is_gpt:
            teacher_config = teacher.config
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
            step = (teacher_config.num_hidden_layers - 1) / (student_config.num_hidden_layers - 1)
            for student_layer, teacher_layer in enumerate(
                int(i * step) for i in range(student_config.num_hidden_layers)
            ):
                student.encoder.layer[student_layer] = teacher.encoder.layer[teacher_layer]

        else:
            teacher_config = teacher.config
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
            step = (teacher_config.n_layer - 1) / (student_config.n_layer - 1)
            for student_layer, teacher_layer in enumerate(int(i * step) for i in range(student_config.n_layer)):
                student.transformer.h[student_layer] = teacher.transformer.h[teacher_layer]
        return student
