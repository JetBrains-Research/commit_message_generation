import wandb
import pandas as pd
import numpy as np
import nltk
import pytorch_lightning as pl

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import defaultdict
from torchmetrics import MetricCollection
from datasets import load_metric

from src.metrics import Accuracy, MRR, EditSimilarity, ExactMatch, ChrF
from src.model.prefix_utils import PrefixAllowedTokens

nltk.download("wordnet")


class GPT2LMHeadModule(pl.LightningModule):
    def __init__(
        self, decoder_name_or_path: str, actual_generation: bool, context_ratio: int, tokenizer: GPT2Tokenizer, **kwargs
    ):
        super().__init__()
        self.actual_generation = actual_generation
        self._tokenizer = tokenizer
        self.save_hyperparameters()

        # use pretrained GPT-2 as decoder
        self.model = GPT2LMHeadModel.from_pretrained(decoder_name_or_path)

        self.context_ratio = context_ratio

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
        self.chrf = load_metric("chrf")
        self.bertscore = load_metric("bertscore")

        self.completion_metrics = MetricCollection(
            {"acc_top1": Accuracy(top_k=1), "acc_top5": Accuracy(top_k=5), "MRR_top5": MRR(top_k=5)}, prefix="test_"
        )

        self.edit_similarity = EditSimilarity()
        self.exact_match = MetricCollection(
            {
                "exact_match@1": ExactMatch(n=1),
                "exact_match@2": ExactMatch(n=2),
                "exact_match@5": ExactMatch(n=5),
            },
            prefix="test_",
        )

        self.table_data = defaultdict(list)

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def forward(self, batch):
        self.examples_count += len(batch["msg_input_ids"])
        return self.model(
            input_ids=batch["msg_input_ids"], attention_mask=batch["msg_attention_mask"], labels=batch["msg_labels"]
        )

    def generate(self, batch):
        self.examples_count += len(batch["msg_input_ids"])
        prefix_fn = PrefixAllowedTokens(
            prefix={i: prefix for i, prefix in enumerate(batch["prefix"])},
            context_len={i: len(msg) for i, msg in enumerate(batch["msg_input_ids"])},
            tokenizer=self._tokenizer,
        )
        return self.model.generate(
            input_ids=batch["msg_input_ids"],
            attention_mask=batch["msg_attention_mask"],
            prefix_allowed_tokens_fn=prefix_fn,
            pad_token_id=self._tokenizer.eos_token_id,
            eos_token_id=198,  # consider \n <EOS> token
            early_stopping=True,
            no_repeat_ngram_size=4,
            num_beams=5,
            min_length=min(5, batch["msg_input_ids"].shape[1]),
            max_length=batch["msg_input_ids"].shape[1] + 25,
        )

    def test_step(self, batch, batch_idx):
        if self.actual_generation:
            return self.actual_generation_step(batch)
        else:
            return self.next_token_metrics_step(batch)

    def actual_generation_step(self, batch):
        # leave only generated part
        gen_sequences = self.generate(batch)[:, batch["msg_input_ids"].shape[1] :]

        # decode tokenized sequences
        decoded_context, decoded_preds, decoded_trg = self.decode_trg(
            batch["msg_input_ids"], gen_sequences, batch["target"]
        )

        # remove prefix from generated and target to compute metrics without it
        decoded_preds = [pred[len(prefix) :].strip("\n") for pred, prefix in zip(decoded_preds, batch["prefix"])]
        decoded_trg = [trg[len(prefix) :] for trg, prefix in zip(decoded_trg, batch["prefix"])]

        # trim by # of generated tokens
        decoded_trg_metrics = []
        for pred, trg in zip(decoded_preds, decoded_trg):
            decoded_trg_metrics.append(" ".join(trg.split()[: max(1, len(pred.split()))]))

        # add data to a little table with examples
        self.table_data["Context"].extend(
            [context + prefix for prefix, context in zip(batch["prefix"], decoded_context)]
        )
        self.table_data["Target (trimmed by # generated tokens)"].extend(decoded_trg_metrics)
        self.table_data["Prediction"].extend(decoded_preds)
        self.table_data["Target (full)"].extend(decoded_trg)

        # add batches to metrics from huggingface datasets
        self.bleu.add_batch(
            predictions=[[token.lower() for token in line.split()] for line in decoded_preds],
            references=[[[token.lower() for token in line.split()]] for line in decoded_trg_metrics],
        )
        self.chrf.add_batch(predictions=decoded_preds, references=[[line] for line in decoded_trg_metrics])
        self.rouge.add_batch(predictions=decoded_preds, references=decoded_trg_metrics)
        self.meteor.add_batch(predictions=decoded_preds, references=decoded_trg_metrics)
        self.bertscore.add_batch(predictions=decoded_preds, references=decoded_trg_metrics)

        # add batches to torchmetrics metrics
        self.edit_similarity(decoded_preds, decoded_trg_metrics)
        self.exact_match(decoded_preds, decoded_trg_metrics)

    def next_token_metrics_step(self, batch):
        scores = self(batch).logits
        return self.completion_metrics(scores, batch["msg_labels"])

    def test_epoch_end(self, outputs):
        if self.actual_generation:
            chrf = self.chrf.compute()
            bleu = self.bleu.compute(smooth=True)
            rouge = self.rouge.compute()
            meteor = self.meteor.compute()
            bertscore = self.bertscore.compute(lang="en")
            edit_sim = self.edit_similarity.compute().item()
            exact_match = self.exact_match.compute()

            df = pd.DataFrame.from_dict(self.table_data)
            table = wandb.Table(dataframe=df)

            test_metrics = {
                "test_examples": table,
                "test_bleu": bleu["bleu"],
                "test_rouge1": rouge["rouge1"].mid.fmeasure,
                "test_rouge2": rouge["rouge2"].mid.fmeasure,
                "test_rougeL": rouge["rougeL"].mid.fmeasure,
                "test_meteor": meteor["meteor"],
                "test_edit_similarity": edit_sim,
                "test_chrf": chrf["score"] / 100,
                "test_bertscore": np.mean(bertscore["f1"]),
            }
            test_metrics.update({key: exact_match[key].cpu().item() for key in exact_match})
            test_metrics = {f"{key}_{self.context_ratio}": test_metrics[key] for key in test_metrics}

            self.logger.experiment.log(test_metrics, step=self.examples_count)
        else:
            test_metrics = self.completion_metrics.compute()
            self.logger.experiment.log(
                {key: test_metrics[key].cpu().item() for key in test_metrics}, step=self.examples_count + 2
            )

    def decode_trg(self, *args):
        return tuple(
            self._tokenizer.batch_decode(arg, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for arg in args
        )
