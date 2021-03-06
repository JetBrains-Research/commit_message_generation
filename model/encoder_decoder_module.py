import pandas as pd
import numpy as np
from collections import defaultdict

import wandb

import torch
import pytorch_lightning as pl
from transformers import EncoderDecoderModel, RobertaTokenizer, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

from metrics import accuracy_MRR
from datasets import load_metric
import nltk

nltk.download('wordnet')


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 encoder_name_or_path: str,
                 decoder_name_or_path: str,
                 unfreeze_after: int,
                 src_tokenizer: RobertaTokenizer,
                 trg_tokenizer: GPT2Tokenizer,
                 num_epochs: int,
                 num_batches: int,
                 **kwargs):
        super().__init__()

        self._unfreeze_after = unfreeze_after
        self._src_tokenizer = src_tokenizer
        self._trg_tokenizer = trg_tokenizer
        self._num_epochs = num_epochs
        self._num_batches = num_batches
        self.learning_rate = learning_rate

        self.save_hyperparameters()

        # use CodeBERTa as encoder and distilGTP-2 as decoder
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(encoder_name_or_path, decoder_name_or_path)

        # cache is currently not supported by EncoderDecoder framework
        self.model.decoder.config.use_cache = False

        # do not tie output embeddings to input embeddings
        self.model.config.tie_word_embeddings = False

        # set decoding params
        self.model.config.decoder_start_token_id = self._trg_tokenizer.bos_token_id
        self.model.config.bos_token_id = self._trg_tokenizer.bos_token_id
        self.model.config.eos_token_id = self._trg_tokenizer.eos_token_id
        self.model.config.pad_token_id = self._trg_tokenizer.pad_token_id
        self.model.config.max_length = 30
        self.model.config.min_length = 2
        self.model.config.no_repeat_ngram_size = 4
        self.model.config.early_stopping = True
        self.model.config.num_beams = 4

        print("\n====MODEL CONFIG====\n")
        print(self.model.config)
        print()

        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")
        self.meteor = load_metric("meteor")

        # to log tables with examples
        self.train_table_data = defaultdict(list)
        self.val_table_data = defaultdict(list)
        self.test_table_data = defaultdict(list)

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def forward(self, batch):
        # transformers assume pad indices to be -100
        # gpt2 has no pad tokens so use attention mask
        self.examples_count += len(batch['diff_input_ids'])

        return self.model(input_ids=batch['diff_input_ids'],
                          attention_mask=batch['diff_attention_mask'],
                          decoder_input_ids=batch['msg_input_ids'],
                          decoder_attention_mask=batch['msg_attention_mask'],
                          labels=batch['msg_input_ids'].where(
                              batch['msg_attention_mask'].type(torch.ByteTensor).to(self.device),
                              torch.tensor(-100, device=self.device)))

    def generate(self, batch):
        return self.model.generate(input_ids=batch['diff_input_ids'],
                                   attention_mask=batch['diff_attention_mask'])

    def on_train_epoch_start(self) -> None:
        # unfreeze everything on certain epoch
        if self.current_epoch == self._unfreeze_after:
            for param in self.model.parameters():
                param.requires_grad = True

    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)[:2]

        # log train examples on every 1000th batch in epoch
        if self.global_step % 1000 == 0:
            with torch.no_grad():
                gen_sequence = self.generate(batch)
                decoded_source, decoded_targets, decoded_preds = self.decode_for_generation(batch['diff_input_ids'],
                                                                                            batch['msg_input_ids'],
                                                                                            gen_sequence)
                # add data to a little table with examples
                self.train_table_data["Source"].extend(decoded_source)
                self.train_table_data["Target"].extend(decoded_targets)
                self.train_table_data["Prediction"].extend(decoded_preds)

        self.logger.experiment.log({"train_loss_step": loss}, step=self.examples_count)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()

        try:
            df = pd.DataFrame.from_dict(self.train_table_data)
            table = wandb.Table(dataframe=df)
            self.train_table_data.clear()

            self.logger.experiment.log({"train_examples": table,
                                        "train_loss_epoch": train_loss_mean}, step=self.examples_count)
        except IndexError:
            self.logger.experiment.log({"train_loss_epoch": train_loss_mean}, step=self.examples_count)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, scores = self(batch)[:2]

        # construct labels by assigning pad tokens to -100
        labels = batch['msg_input_ids'].where(batch['msg_attention_mask'].type(torch.ByteTensor).to(self.device),
                                              torch.tensor(-100, device=self.device))

        # calculate metrics
        acc_top1, acc_top5, MRR_top5 = accuracy_MRR(scores, labels, top_k=5, shift=True)

        # get top k predictions for each token in each batch
        _, top_k_predictions = torch.topk(scores, k=5)

        # assign target pad tokens idx to pad_token_id to avoid logging them in table
        top_k_predictions[labels == -100, :] = self._trg_tokenizer.pad_token_id

        # decode sources, targets and top-k "predictions"
        decoded_sources, decoded_targets, decoded_top_k_preds = self.decode_for_metrics(batch['diff_input_ids'],
                                                                                        batch['msg_input_ids'],
                                                                                        top_k_predictions)

        # add data to a little table with examples
        self.val_table_data["Source"].extend(decoded_sources)
        self.val_table_data["Target"].extend(decoded_targets)
        for i in range(5):
            self.val_table_data[f"Top {i + 1}"].extend(decoded_top_k_preds[i])

        return {"val_loss": loss, "acc_top1": acc_top1, "acc_top5": acc_top5, "MRR_top5": MRR_top5}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_acc_top1 = np.mean([x["acc_top1"] for x in outputs])
        val_acc_top5 = np.mean([x["acc_top5"] for x in outputs])
        val_MRR_top5 = np.mean([x["MRR_top5"] for x in outputs])

        df = pd.DataFrame.from_dict(self.val_table_data)
        table = wandb.Table(dataframe=df)
        self.val_table_data.clear()

        self.logger.experiment.log({"val_examples": table,
                                    "val_loss": val_loss_mean,
                                    "val_acc_top1": val_acc_top1,
                                    "val_acc_top5": val_acc_top5,
                                    "val_MRR_top5": val_MRR_top5},
                                   step=self.examples_count)
        # needed for ModelCheckpoint
        self.log('val_MRR_top5', val_MRR_top5, on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def test_step(self, batch, batch_idx):
        # generate
        gen_sequence = self.generate(batch)
        decoded_source, decoded_targets, decoded_preds = self.decode_for_generation(batch['diff_input_ids'],
                                                                                    batch['msg_input_ids'],
                                                                                    gen_sequence)

        # add data to a little table with examples
        self.test_table_data["Source"].extend(decoded_source)
        self.test_table_data["Target"].extend(decoded_targets)
        self.test_table_data["Prediction"].extend(decoded_preds)

        # add batches to metrics
        self.bleu.add_batch(predictions=[line.split() for line in decoded_preds],
                            references=[[line.split()] for line in decoded_targets])
        self.rouge.add_batch(predictions=decoded_preds, references=decoded_targets)
        self.meteor.add_batch(predictions=decoded_preds, references=decoded_targets)

    def test_epoch_end(self, outputs):
        bleu = self.bleu.compute()
        rouge = self.rouge.compute()
        meteor = self.meteor.compute()

        df = pd.DataFrame.from_dict(self.test_table_data)
        table = wandb.Table(dataframe=df)
        self.test_table_data.clear()

        self.logger.experiment.log({"test_examples": table,
                                    "test_bleu": bleu["bleu"],
                                    "test_rouge1": rouge["rouge1"].mid.fmeasure,
                                    "test_rouge2": rouge["rouge2"].mid.fmeasure,
                                    "test_rougeL": rouge["rougeL"].mid.fmeasure,
                                    "test_meteor": meteor["meteor"]}, step=self.examples_count)

    def decode_for_generation(self, sources, targets, preds):
        decoded_sources = self._src_tokenizer.batch_decode(sources, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False)
        decoded_targets = self._trg_tokenizer.batch_decode(targets, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False)
        decoded_preds = self._trg_tokenizer.batch_decode(preds, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False)
        return decoded_sources, decoded_targets, decoded_preds

    def decode_for_metrics(self, sources, targets, top_k_preds):
        decoded_targets = self._trg_tokenizer.batch_decode(targets, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False)
        decoded_sources = self._src_tokenizer.batch_decode(sources, skip_special_tokens=True,
                                                           clean_up_tokenization_spaces=False)
        decoded_top_k_preds = [self._trg_tokenizer.batch_decode(top_k_preds[:, :, i], skip_special_tokens=True,
                                                                clean_up_tokenization_spaces=False) for i in range(5)]

        return decoded_sources, decoded_targets, decoded_top_k_preds

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer, self._num_batches * 5,
                                                                  self._num_epochs * self._num_batches),
                     'interval': 'step',
                     'frequency': 1}
        return [optimizer], [scheduler]
