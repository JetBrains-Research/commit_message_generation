import pytorch_lightning as pl

from collections import defaultdict
import pandas as pd
import numpy as np

import torch

from transformers import EncoderDecoderModel, RobertaModel, RobertaConfig, GPT2LMHeadModel, GPT2Config, \
    RobertaTokenizer, GPT2Tokenizer

import wandb

from metrics import accuracy_MRR
from datasets import load_metric
import nltk

nltk.download('wordnet')


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self,
                 num_layers_encoder: int,
                 num_layers_decoder: int,
                 actual_generation: bool,
                 src_tokenizer: RobertaTokenizer,
                 trg_tokenizer: GPT2Tokenizer,
                 **kwargs):
        super().__init__()

        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self.actual_generation = actual_generation
        self._src_tokenizer = src_tokenizer
        self._trg_tokenizer = trg_tokenizer
        self.save_hyperparameters()

        # use randomly initialized RoBERTa as encoder
        encoder_config = RobertaConfig()
        encoder_config.num_hidden_layers = self.num_layers_encoder
        encoder = RobertaModel(config=encoder_config)
        encoder.resize_token_embeddings(len(self._src_tokenizer))

        # use randomly initialized GPT-2 as decoder
        decoder_config = GPT2Config()
        decoder_config.n_layer = self.num_layers_decoder
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        gpt = GPT2LMHeadModel(config=decoder_config)

        self.model = EncoderDecoderModel(encoder=encoder, decoder=gpt)

        # cache is currently not supported by EncoderDecoder framework
        self.model.decoder.config.use_cache = False

        # do not tie output embeddings to input embeddings
        self.model.config.tie_word_embeddings = False

        # set decoding params
        self.model.config.no_repeat_ngram_size = 4
        self.model.config.early_stopping = True
        self.model.config.num_beams = 4

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
        self.examples_count += len(batch['diff_input_ids'])

        return self.model(input_ids=batch['diff_input_ids'],
                          attention_mask=batch['diff_attention_mask'],
                          decoder_input_ids=batch['msg_input_ids'],
                          decoder_attention_mask=batch['msg_attention_mask'],
                          labels=batch['msg_input_ids'].where(
                              batch['msg_attention_mask'].type(torch.ByteTensor).to(self.device),
                              torch.tensor(-100, device=self.device)))

    def test_step(self, batch, batch_idx):
        if self.actual_generation:
            return self.actual_generation_step(batch)
        else:
            return self.next_token_metrics_step(batch)

    def actual_generation_step(self, batch):
        preds = self.model.generate(input_ids=batch['diff_input_ids'],
                                    attention_mask=batch['diff_attention_mask'],
                                    decoder_input_ids=batch['generation_input_ids'],
                                    decoder_attention_mask=batch['generation_attention_mask'],
                                    min_length=batch['generation_input_ids'].shape[1] + 5,
                                    max_length=batch['generation_input_ids'].shape[1] + 5)

        # consider only 5 generated tokens
        preds[:, :-5] = self._trg_tokenizer.pad_token_id

        # assign -100 in message to pad_token_id to be able to decode but avoid logging them in table
        message_completion = batch['generation_labels'].detach().clone()
        message_completion[message_completion == -100] = self._trg_tokenizer.pad_token_id

        # decode generated sequences and targets into strings
        decoded_targets_whole_seq, decoded_targets_completion, decoded_preds = \
            self.decode_for_actual_generation(batch['msg_input_ids'],
                                              message_completion,
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

        # construct labels by assigning pad tokens to -100
        labels = batch['msg_input_ids'].where(batch['msg_attention_mask'].type(torch.ByteTensor).to(self.device),
                                              torch.tensor(-100, device=self.device))

        # calculate metrics
        acc_top1, acc_top5, MRR_top5 = accuracy_MRR(scores, labels, top_k=5, shift=True)

        # get top k predictions for each token in each batch
        _, top_k_predictions = torch.topk(scores, k=5)

        # assign target pad tokens idx to pad_token_id to avoid logging them in table
        top_k_predictions[labels == -100, :] = self._trg_tokenizer.pad_token_id

        # decode top k predictions and targets
        preds, source, targets = self.decode_for_metrics(top_k_predictions,
                                                         batch['diff_input_ids'], batch['msg_input_ids'])

        # add data to a little table with examples
        self.table_data["Source"].extend(source)
        self.table_data["Target"].extend(targets)
        for i in range(5):
            self.table_data[f"Top {i + 1}"].extend(preds[i])

        return {"acc_top1": acc_top1, "acc_top5": acc_top5, "MRR_top5": MRR_top5}

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
        decoded_preds = self._trg_tokenizer.batch_decode(preds, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False)
        decoded_targets_whole_seq = self._trg_tokenizer.batch_decode(targets_whole_seq, skip_special_tokens=True,
                                                                     clean_up_tokenization_spaces=False)
        decoded_targets_completion = self._trg_tokenizer.batch_decode(targets_completion, skip_special_tokens=True,
                                                                      clean_up_tokenization_spaces=False)
        return decoded_targets_whole_seq, decoded_targets_completion, decoded_preds

    def decode_for_metrics(self, top_k_preds, source, target):
        # decoded preds and targets
        targets = self._trg_tokenizer.batch_decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        source = self._src_tokenizer.batch_decode(source, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        preds = [self._trg_tokenizer.batch_decode(top_k_preds[:, :, i], skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False) for i in range(5)]

        return preds, source, targets
