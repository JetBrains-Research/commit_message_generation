import numpy as np  # type: ignore
from copy import copy
from typing import Optional
import torch
import pytorch_lightning as pl
from transformers import EncoderDecoderModel, RobertaModel, RobertaConfig, GPT2LMHeadModel, GPT2Config  # type: ignore
from transformers import RobertaTokenizer, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup  # type: ignore
from metrics import accuracy_MRR  # type: ignore


class EncoderDecoderModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        src_tokenizer: RobertaTokenizer,
        trg_tokenizer: GPT2Tokenizer,
        num_epochs: int,
        num_batches: int,
        num_gpus: int,
        num_layers_encoder: Optional[int] = None,
        num_layers_decoder: Optional[int] = None,
        encoder_name_or_path: Optional[str] = None,
        decoder_name_or_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        self._src_tokenizer = src_tokenizer
        self._trg_tokenizer = trg_tokenizer
        self._num_epochs = num_epochs
        self._num_batches = num_batches
        self._num_gpus = num_gpus
        self.learning_rate = learning_rate

        self.save_hyperparameters()

        if encoder_name_or_path is not None and decoder_name_or_path is not None:
            # use pretrained RoBERTa as encoder
            encoder = RobertaModel.from_pretrained(encoder_name_or_path)
            # resize embeddings to match vocabulary size
            encoder.resize_token_embeddings(len(self._src_tokenizer))
            # remove layers if necessary
            if num_layers_encoder is not None and num_layers_encoder < encoder.config.num_hidden_layers:
                encoder = EncoderDecoderModule.remove_layers_from_model(encoder, num_layers_encoder, is_gpt=False)

            # use pretrained GPT-2 as decoder
            config = GPT2Config.from_pretrained(decoder_name_or_path)
            config.is_decoder = True
            config.add_cross_attention = True
            decoder = GPT2LMHeadModel.from_pretrained(decoder_name_or_path, config=config)
            # remove layers if necessary
            if num_layers_decoder is not None and num_layers_decoder < decoder.config.n_layer:
                decoder = EncoderDecoderModule.remove_layers_from_model(decoder, num_layers_decoder, is_gpt=True)

        elif num_layers_decoder is not None and num_layers_encoder is not None:
            # use randomly initialized RoBERTa as encoder
            encoder_config = RobertaConfig()
            encoder_config.num_hidden_layers = num_layers_encoder
            encoder = RobertaModel(config=encoder_config)
            # resize embeddings to match vocabulary size
            encoder.resize_token_embeddings(len(self._src_tokenizer))

            # use randomly initialized GPT-2 as decoder
            decoder_config = GPT2Config()
            decoder_config.n_layer = num_layers_decoder
            decoder_config.is_decoder = True
            decoder_config.add_cross_attention = True
            decoder = GPT2LMHeadModel(config=decoder_config)
        else:
            raise ValueError(
                "You have to specify either num_layers for training from scratch \
                                          or paths for loading pretrained models"
            )

        self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

        # cache is currently not supported by EncoderDecoder framework
        self.model.decoder.config.use_cache = False

        # do not tie output embeddings to input embeddings
        self.model.config.tie_word_embeddings = False

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

    def training_step(self, batch, batch_idx):
        self.examples_count += len(batch["diff_input_ids"])
        loss, logits = self(batch)[:2]
        self.logger.experiment.log({"train_loss_step": loss}, step=self.examples_count)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.log({"train_loss_epoch": train_loss_mean}, step=self.examples_count)

    def metrics_step(self, batch):
        """
        Logic for validation & testing steps:
        1) Calculate accuracy@1, accuracy@5 and MRR@5 from model_utils output and labels
        """
        loss, scores = self(batch)[:2]
        # calculate metrics
        acc_top1, acc_top5, MRR_top5 = accuracy_MRR(scores, batch["msg_labels"], top_k=5, shift=True)
        return {"loss": loss, "acc_top1": acc_top1, "acc_top5": acc_top5, "MRR_top5": MRR_top5}

    def metrics_epoch_end(self, outputs, stage):
        """
        Logic for validation & testing epoch end:
        1) Calculate final accuracy@1, accuracy@5, MRR@5
        2) (in val stage only) Aggregate loss and log metric(s) for ModelCheckpoint
        3) Log everything in wandb.
        """
        acc_top1 = np.mean([x["acc_top1"] for x in outputs])
        acc_top5 = np.mean([x["acc_top5"] for x in outputs])
        MRR_top5 = np.mean([x["MRR_top5"] for x in outputs])

        results = {f"{stage}_acc_top1": acc_top1, f"{stage}_acc_top5": acc_top5, f"{stage}_MRR_top5": MRR_top5}

        if stage == "val":
            loss = torch.stack([x["loss"] for x in outputs]).mean()
            results["val_loss"] = loss
            # needed for ModelCheckpoint
            self.log("val_MRR_top5", MRR_top5, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.logger.experiment.log(results, step=self.examples_count)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.metrics_step(batch)

    def validation_epoch_end(self, outputs):
        self.metrics_epoch_end(outputs, stage="val")

    def test_step(self, batch, batch_idx):
        return self.metrics_step(batch)

    def test_epoch_end(self, outputs):
        self.metrics_epoch_end(outputs, stage="test")

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
