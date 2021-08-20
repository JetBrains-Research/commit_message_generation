import wandb
import pandas as pd
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
    def __init__(self, decoder_name_or_path: str, actual_generation: bool, tokenizer: GPT2Tokenizer, **kwargs):
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
        self.chrf = ChrF()

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
            min_length=0,
            max_length=batch["msg_input_ids"].shape[1] + 25,
        )

    def test_step(self, batch, batch_idx):
        if self.actual_generation:
            return self.actual_generation_step(batch)
        else:
            return self.next_token_metrics_step(batch)

    def actual_generation_step(self, batch):
        # leave only generated part
        gen_sequences = self.generate(batch)[:, batch["msg_input_ids"].shape[1]:]
        # trim by # of generated tokens
        trg_sequences = batch["target"][:, : gen_sequences.shape[1]]

        # decode tokenized sequences
        decoded_context, decoded_preds, decoded_trg = self.decode_trg(
            batch["msg_input_ids"], gen_sequences, trg_sequences
        )

        # remove prefix from generated and target to compute metrics without it
        decoded_preds = [
            pred[len(prefix):].strip("\n") for pred, prefix in zip(decoded_preds, batch["prefix"])
        ]
        decoded_trg = [trg[len(prefix):] for trg, prefix in zip(decoded_trg, batch["prefix"])]

        # add data to a little table with examples
        self.table_data["Context"].extend(
            [context.strip() + " " + prefix.strip() for context, prefix in zip(decoded_context, batch["prefix"])]
        )
        self.table_data["Target"].extend(decoded_trg)
        self.table_data["Prediction"].extend(decoded_preds)

        # add batches to metrics from huggingface datasets
        self.bleu.add_batch(
            predictions=[line.split() for line in decoded_preds],
            references=[[line.split()] for line in decoded_trg],
        )
        self.chrf.add_batch(
            predictions=[line.split() for line in decoded_preds], references=[line.split() for line in decoded_trg]
        )
        self.rouge.add_batch(predictions=decoded_preds, references=decoded_trg)
        self.meteor.add_batch(predictions=decoded_preds, references=decoded_trg)

        # add batches to torchmetrics metrics
        self.edit_similarity(decoded_preds, decoded_trg)
        self.exact_match(decoded_preds, decoded_trg)

        # compute and log current values of torchmetrics metrics
        edit_sim_step = self.edit_similarity.compute()
        exact_match_step = self.exact_match.compute()
        test_metrics_step = {
            "test_edit_similarity_step": edit_sim_step,
        }
        test_metrics_step.update({key: exact_match_step[key].cpu().item() for key in exact_match_step})
        self.logger.experiment.log(test_metrics_step, step=self.examples_count)

    def next_token_metrics_step(self, batch):
        scores = self(batch).logits
        return self.completion_metrics(scores, batch["msg_labels"])

    def test_epoch_end(self, outputs):
        if self.actual_generation:
            chrf = self.chrf.compute()
            bleu = self.bleu.compute()
            rouge = self.rouge.compute()
            meteor = self.meteor.compute()
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
                "test_chrf": chrf["chrf"],
            }
            test_metrics.update({key: exact_match[key].cpu().item() for key in exact_match})

            self.logger.experiment.log(test_metrics, step=self.examples_count)
        else:
            test_metrics = self.completion_metrics.compute()
            self.logger.experiment.log(
                {key: test_metrics[key].cpu().item() for key in test_metrics}, step=self.examples_count
            )

    def decode_trg(self, *args):
        return tuple(
            self._tokenizer.batch_decode(arg, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for arg in args
        )
