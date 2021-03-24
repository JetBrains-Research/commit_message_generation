import pandas as pd
import numpy as np
from collections import defaultdict

import wandb

import torch
import pytorch_lightning as pl
from transformers import EncoderDecoderModel, RobertaModel, RobertaConfig, GPT2LMHeadModel, GPT2Config, \
    RobertaTokenizer, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

from metrics import accuracy_MRR
from datasets import load_metric
import nltk

nltk.download('wordnet')


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 num_layers_encoder: int,
                 num_layers_decoder: int,
                 src_tokenizer: RobertaTokenizer,
                 trg_tokenizer: GPT2Tokenizer,
                 num_epochs: int,
                 num_batches: int,
                 **kwargs):
        super().__init__()

        self._src_tokenizer = src_tokenizer
        self._trg_tokenizer = trg_tokenizer
        self._num_epochs = num_epochs
        self._num_batches = num_batches
        self.learning_rate = learning_rate

        self.save_hyperparameters()

        # use randomly initialized RoBERTa as encoder
        encoder_config = RobertaConfig()
        encoder_config.num_hidden_layers = num_layers_encoder
        encoder = RobertaModel(config=encoder_config)

        # resize embeddings to match CodeBERT vocab
        encoder.resize_token_embeddings(len(self._src_tokenizer))

        # use randomly initialized GPT-2 as decoder
        decoder_config = GPT2Config()
        decoder_config.n_layer = num_layers_decoder
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder = GPT2LMHeadModel(config=decoder_config)

        self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

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
        return self.model(input_ids=batch['diff_input_ids'],
                          attention_mask=batch['diff_attention_mask'],
                          decoder_input_ids=batch['msg_input_ids'],
                          decoder_attention_mask=batch['msg_attention_mask'],
                          labels=batch['msg_labels'])

    def generate(self, batch):
        return self.model.generate(input_ids=batch['diff_input_ids'],
                                   attention_mask=batch['diff_attention_mask'],
                                   decoder_input_ids=batch['generation_input_ids'],
                                   decoder_attention_mask=batch['generation_attention_mask'],
                                   max_length=batch['msg_input_ids'].shape[1])

    def training_step(self, batch, batch_idx):
        self.examples_count += len(batch['diff_input_ids'])
        loss, logits = self(batch)[:2]

        # log train examples on every 1000th batch in epoch
        if self.global_step % 1000 == 0:
            with torch.no_grad():
                gen_sequence = self.generate(batch)
                gen_input_len = batch['generation_input_ids'].shape[1]
                gen_sequence = [i[gen_input_len:] for i in gen_sequence.tolist()]  # leave only generated part

                targets_no_history = batch['msg_input_ids'].detach().clone().to(self.device)
                targets_no_history[batch['msg_labels'] == -100] = self._trg_tokenizer.pad_token_id

                decoded_source = self.decode_src(batch['diff_input_ids'])[0]
                decoded_targets_no_history, decoded_history, decoded_preds = \
                    self.decode_trg(targets_no_history,
                                    batch['generation_input_ids'],
                                    gen_sequence)

                # add data to a little table with examples
                self.train_table_data["Diff"].extend(decoded_source)
                self.train_table_data["History (generation input)"].extend(decoded_history)
                self.train_table_data["Target"].extend(decoded_targets_no_history)
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

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            return self.generate_step(batch)
        elif dataloader_idx == 1:
            return self.metrics_step(batch)

    def validation_epoch_end(self, outputs):
        jiang_outputs = outputs[0]
        github_outputs = outputs[1]

        val_jiang_loss_mean = torch.stack([x["val_loss"] for x in jiang_outputs]).mean()
        bleu = self.bleu.compute()
        rouge = self.rouge.compute()
        meteor = self.meteor.compute()

        val_github_loss_mean = torch.stack([x["val_loss"] for x in github_outputs]).mean()
        val_acc_top1 = np.mean([x["acc_top1"] for x in github_outputs])
        val_acc_top5 = np.mean([x["acc_top5"] for x in github_outputs])
        val_MRR_top5 = np.mean([x["MRR_top5"] for x in github_outputs])

        df = pd.DataFrame.from_dict(self.val_table_data)
        table = wandb.Table(dataframe=df)
        self.val_table_data.clear()

        self.logger.experiment.log({"val_examples": table,
                                    "val_loss_jiang": val_jiang_loss_mean,
                                    "val_bleu": bleu["bleu"],
                                    "val_rouge1": rouge["rouge1"].mid.fmeasure,
                                    "val_rouge2": rouge["rouge2"].mid.fmeasure,
                                    "val_rougeL": rouge["rougeL"].mid.fmeasure,
                                    "val_meteor": meteor["meteor"],
                                    "val_loss_github": val_github_loss_mean,
                                    "val_acc_top1": val_acc_top1,
                                    "val_acc_top5": val_acc_top5,
                                    "val_MRR_top5": val_MRR_top5},
                                   step=self.examples_count)
        # needed for ModelCheckpoint
        self.log('val_MRR_top5', val_MRR_top5, on_step=False, on_epoch=True, prog_bar=True, logger=False)

    def test_step(self, batch, batch_idx):
        # generate
        gen_sequence = self.generate(batch)
        gen_input_len = batch['generation_input_ids'].shape[1]
        gen_sequence = [i[gen_input_len:] for i in gen_sequence.tolist()]  # leave only generated part

        targets_no_history = batch['msg_input_ids'].detach().clone().to(self.device)
        targets_no_history[batch['msg_labels'] == -100] = self._trg_tokenizer.pad_token_id

        decoded_source = self.decode_src(batch['diff_input_ids'])[0]
        decoded_targets_no_history, decoded_history, decoded_preds = \
            self.decode_trg(targets_no_history,
                            batch['generation_input_ids'],
                            gen_sequence)

        # add data to a little table with examples
        self.test_table_data["Diff"].extend(decoded_source)
        self.test_table_data["History (generation input)"].extend(decoded_history)
        self.test_table_data["Target"].extend(decoded_targets_no_history)
        self.test_table_data["Prediction"].extend(decoded_preds)

        # add batches to metrics
        self.bleu.add_batch(predictions=[line.split() for line in decoded_preds],
                            references=[[line.split()] for line in decoded_targets_no_history])
        self.rouge.add_batch(predictions=decoded_preds, references=decoded_targets_no_history)
        self.meteor.add_batch(predictions=decoded_preds, references=decoded_targets_no_history)

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

    def generate_step(self, batch):
        """
        val step for jiang dataloader - generate
        """
        loss, scores = self(batch)[:2]
        # generate
        gen_sequence = self.generate(batch)
        gen_input_len = batch['generation_input_ids'].shape[1]
        gen_sequence = [i[gen_input_len:] for i in gen_sequence.tolist()]  # leave only generated part

        targets_no_history = batch['msg_input_ids'].detach().clone().to(self.device)
        targets_no_history[batch['msg_labels'] == -100] = self._trg_tokenizer.pad_token_id

        decoded_source = self.decode_src(batch['diff_input_ids'])[0]
        decoded_targets_no_history, decoded_history, decoded_preds = \
            self.decode_trg(targets_no_history,
                            batch['generation_input_ids'],
                            gen_sequence)

        # add data to a little table with examples
        self.val_table_data["Diff"].extend(decoded_source)
        self.val_table_data["History (generation input)"].extend(decoded_history)
        self.val_table_data["Target"].extend(decoded_targets_no_history)
        self.val_table_data["Prediction"].extend(decoded_preds)

        # add batches to metrics
        self.bleu.add_batch(predictions=[line.split() for line in decoded_preds],
                            references=[[line.split()] for line in decoded_targets_no_history])
        self.rouge.add_batch(predictions=decoded_preds, references=decoded_targets_no_history)
        self.meteor.add_batch(predictions=decoded_preds, references=decoded_targets_no_history)

        return {"val_loss": loss}

    def metrics_step(self, batch):
        """
        val step for github dataloader - calculate metrics for completion
        """
        # breaks here
        loss, scores = self(batch)[:2]

        # calculate metrics
        acc_top1, acc_top5, MRR_top5 = accuracy_MRR(scores, batch['msg_labels'], top_k=5, shift=True)

        return {"val_loss": loss, "acc_top1": acc_top1, "acc_top5": acc_top5, "MRR_top5": MRR_top5}

    def decode_src(self, *args):
        return tuple(self._src_tokenizer.batch_decode(arg, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False) for arg in args)

    def decode_trg(self, *args):
        return tuple(self._trg_tokenizer.batch_decode(arg, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False) for arg in args)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer, self._num_batches * 5,
                                                                  self._num_epochs * self._num_batches),
                     'interval': 'step',
                     'frequency': 1}
        return [optimizer], [scheduler]
