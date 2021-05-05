import wandb

from collections import defaultdict
import pandas as pd
import numpy as np

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import pytorch_lightning as pl

from metrics import accuracy_MRR
from datasets import load_metric
import nltk

nltk.download('wordnet')


class GPT2LMHeadModule(pl.LightningModule):
    def __init__(self,
                 decoder_name_or_path: str,
                 actual_generation: bool,
                 tokenizer: GPT2Tokenizer,
                 **kwargs):
        super().__init__()
        self.actual_generation = actual_generation
        self._tokenizer = tokenizer
        self.save_hyperparameters()

        # use pretrained GPT-2 as decoder
        self.model = GPT2LMHeadModel.from_pretrained(decoder_name_or_path)

        # generating params
        self.model.config.no_repeat_ngram_size = 4
        self.model.config.early_stopping = True
        self.model.config.num_beams = 4
        self.model.config.pad_token_id = self._tokenizer.eos_token_id

        print("\n====MODEL CONFIG====\n")
        print(self.model.config)
        print()

        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")
        self.meteor = load_metric("meteor")

        self.table_data = defaultdict(list)

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def forward(self, batch):
        self.examples_count += len(batch['msg_input_ids'])
        return self.model(input_ids=batch['msg_input_ids'],
                          attention_mask=batch['msg_attention_mask'],
                          labels=batch['msg_labels'])

    def generate(self, batch):
        return self.model.generate(input_ids=batch['generation_input_ids'],
                                   attention_mask=batch['generation_attention_mask'],
                                   max_length=batch['msg_input_ids'].shape[1])

    def actual_generation_step(self, batch):
        gen_sequence = self.generate(batch)
        gen_input_len = batch['generation_input_ids'].shape[1]
        gen_sequence = [i[gen_input_len:] for i in gen_sequence.tolist()]  # leave only generated part

        targets_no_history = batch['msg_input_ids'].detach().clone().to(self.device)
        targets_no_history[batch['msg_labels'] == -100] = self._tokenizer.pad_token_id

        decoded_targets_no_history, decoded_history, decoded_preds = \
            self.decode_trg(targets_no_history,
                            batch['generation_input_ids'],
                            gen_sequence)

        # add data to a little table with examples
        self.table_data["History (generation input)"].extend(decoded_history)
        self.table_data["Target"].extend(decoded_targets_no_history)
        self.table_data["Prediction"].extend(decoded_preds)

        # add batches to metrics
        self.bleu.add_batch(predictions=[line.split() for line in decoded_preds],
                            references=[[line.split()] for line in decoded_targets_no_history])
        self.rouge.add_batch(predictions=decoded_preds, references=decoded_targets_no_history)
        self.meteor.add_batch(predictions=decoded_preds, references=decoded_targets_no_history)

    def next_token_metrics_step(self, batch):
        print(batch['msg_input_ids'].shape)
        scores = self(batch).logits
        # calculate metrics
        acc_top1, acc_top5, MRR_top5 = accuracy_MRR(scores, batch['msg_labels'], top_k=5, shift=True)

        return {"acc_top1": acc_top1, "acc_top5": acc_top5, "MRR_top5": MRR_top5}

    def test_step(self, batch, batch_idx):
        if self.actual_generation:
            return self.actual_generation_step(batch)
        else:
            return self.next_token_metrics_step(batch)

    def test_epoch_end(self, outputs):
        if self.actual_generation:
            bleu = self.bleu.compute()
            rouge = self.rouge.compute()
            meteor = self.meteor.compute()

            df = pd.DataFrame.from_dict(self.table_data)
            table = wandb.Table(dataframe=df)

            self.logger.experiment.log({"test_examples": table,
                                        "test_bleu": bleu["bleu"],
                                        "test_rouge1": rouge["rouge1"].mid.fmeasure,
                                        "test_rouge2": rouge["rouge2"].mid.fmeasure,
                                        "test_rougeL": rouge["rougeL"].mid.fmeasure,
                                        "test_meteor": meteor["meteor"]}, step=self.examples_count)
        else:
            df = pd.DataFrame.from_dict({'acc_top1': [x["acc_top1"] for x in outputs],
                                         'acc_top5': [x["acc_top5"] for x in outputs],
                                         'MRR_top5': [x["MRR_top5"] for x in outputs]})
            wandb.Table.MAX_ROWS = 15000
            table = wandb.Table(dataframe=df)
            self.logger.experiment.log({"test_metrics_for_CI": table,
                                        "test_acc_top1": np.mean([x["acc_top1"] for x in outputs]),
                                        "test_acc_top5": np.mean([x["acc_top5"] for x in outputs]),
                                        "test_MRR_top5": np.mean([x["MRR_top5"] for x in outputs])}, step=9000001)

    def decode_trg(self, *args):
        return tuple(self._tokenizer.batch_decode(arg, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False) for arg in args)
