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
        self.examples_count += len(batch['input_ids'])
        return self.model(input_ids=batch['input_ids'],
                          attention_mask=batch['attention_mask'],
                          labels=batch['labels'])

    def actual_generation_step(self, batch):
        preds = self.model.generate(input_ids=batch['generation_input_ids'],
                                    attention_mask=batch['generation_attention_mask'],
                                    min_length=batch['generation_input_ids'].shape[1] + 5,
                                    max_length=batch['generation_input_ids'].shape[1] + 5)

        # consider only 5 generated tokens
        preds[:, :-5] = self._tokenizer.pad_token_id

        # assign history & padding tokens to pad_token_id to avoid logging them in table
        targets_whole_seq = batch['labels'].detach().clone()
        targets_whole_seq[targets_whole_seq == -100] = self._tokenizer.pad_token_id
        targets_completion = batch['generation_labels'].detach().clone()
        targets_completion[targets_completion == -100] = self._tokenizer.pad_token_id

        # decode generated sequences and targets into strings
        decoded_targets_whole_seq, decoded_targets_completion, decoded_preds = \
            self.decode_for_actual_generation(targets_whole_seq,
                                              targets_completion,
                                              preds)

        # add data to a little table with examples
        self.table_data["Target (whole seq)"].extend(decoded_targets_whole_seq)
        self.table_data["Target (to complete)"].extend(decoded_targets_completion)
        self.table_data["Generated completion"].extend(decoded_preds)

        # add batches to metrics
        self.bleu.add_batch(predictions=[line.split() for line in decoded_preds],
                            references=[[line.split()] for line in decoded_targets_completion])
        self.rouge.add_batch(predictions=decoded_preds, references=decoded_targets_completion)
        self.meteor.add_batch(predictions=decoded_preds, references=decoded_targets_completion)

    def next_token_metrics_step(self, batch):
        scores = self(batch).logits
        # calculate metrics
        acc_top1, acc_top5, MRR_top5 = accuracy_MRR(scores, batch['labels'], top_k=5, shift=True)

        # get top k predictions for each token in each batch
        _, top_k_predictions = torch.topk(scores, k=5)

        # assign history & padding tokens to pad_token_id to avoid logging them in table
        top_k_predictions[batch['labels'] == -100, :] = self._tokenizer.pad_token_id
        targets_no_history = batch['labels']
        targets_no_history[targets_no_history == -100] = self._tokenizer.pad_token_id

        # decode top k predictions and targets
        decoded_preds, decoded_targets_no_history = \
            self.decode_for_metrics(top_k_predictions,
                                    targets_no_history)

        # add data to a little table with examples
        self.table_data["Target (no history)"].extend(decoded_targets_no_history)
        for i in range(5):
            self.table_data[f"Top {i + 1}"].extend(decoded_preds[i])

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
            test_acc_top1 = np.mean([x["acc_top1"] for x in outputs])
            test_acc_top5 = np.mean([x["acc_top5"] for x in outputs])
            test_MRR_top5 = np.mean([x["MRR_top5"] for x in outputs])

            df = pd.DataFrame.from_dict(self.table_data)
            table = wandb.Table(dataframe=df)

            self.logger.experiment.log({"test_examples": table,
                                        "test_acc_top1": test_acc_top1,
                                        "test_acc_top5": test_acc_top5,
                                        "test_MRR_top5": test_MRR_top5},
                                       step=self.examples_count)

    def decode_for_actual_generation(self, targets_whole_seq, targets_completion, preds):
        decoded_preds = self._tokenizer.batch_decode(preds, skip_special_tokens=True,
                                                     clean_up_tokenization_spaces=False)
        decoded_targets_whole_seq = self._tokenizer.batch_decode(targets_whole_seq, skip_special_tokens=True,
                                                                         clean_up_tokenization_spaces=False)
        decoded_targets_completion = self._tokenizer.batch_decode(targets_completion, skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=False)
        return decoded_targets_whole_seq, decoded_targets_completion, decoded_preds

    def decode_for_metrics(self, top_k_preds, targets_no_history):
        decoded_preds = [self._tokenizer.batch_decode(top_k_preds[:, :, i], skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False) for i in range(5)]

        decoded_targets_no_history = self._tokenizer.batch_decode(targets_no_history, skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=False)

        return decoded_preds, decoded_targets_no_history

