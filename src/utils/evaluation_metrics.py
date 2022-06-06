from typing import Dict, List, Optional

import torch
from datasets import load_metric
from torch import Tensor
from torchmetrics import MetricCollection

from src.metrics import MRR, Accuracy, BLEUNorm, EditSimilarity, ExactMatch, LogMNEXT


class EvaluationMetrics:
    """This class is used to compute all evaluation metrics for commit message completion task.

    Currently, it includes the following:

    * string similarity metrics: BLEU, B-NORM, ROUGE, METEOR, LogM-Next, ChrF
    * completion metrics: Accuracy@k, MRR@k, Exact Match@k, Edit Similarity

    Accuracy@k and MRR@k are calculated on raw model output (tensors), all other metrics are calculated on
    generated and decoded strings.

    Args:
        do_tensors: True to compute Accuracy@k and MRR@k and False otherwise.
        do_strings: True to compute string similarity metrics and False otherwise.
        n: if an integer is given, ExactMatch metrics will be computed for first n tokens. Otherwise, it is computed for the whole sequences.
    """

    def __init__(self, do_strings: bool, do_tensors: bool, n: Optional[int] = None, prefix: Optional[str] = None):
        if do_tensors:
            self.tensors_metrics = MetricCollection(
                {"acc_top1": Accuracy(top_k=1), "acc_top5": Accuracy(top_k=5), "MRR_top5": MRR(top_k=5)},
            )
        else:
            self.tensors_metrics = None

        if do_strings:
            self.datasets_metrics = {
                "b_norm": BLEUNorm(),
                "bleu": load_metric("bleu"),
                "rouge": load_metric("rouge"),
                "meteor": load_metric("meteor"),
                "chrf": load_metric("chrf"),
            }
            self.strings_metrics = MetricCollection(
                {
                    "exact_match": ExactMatch(n=n),
                    "edit_similarity": EditSimilarity(),
                    "log_mnext": LogMNEXT(),
                }
            )
        else:
            self.datasets_metrics = None
            self.strings_metrics = None

        self.prefix = prefix

    def add_batch(
        self,
        predictions: Optional[List[str]] = None,
        references: Optional[List[str]] = None,
        predictions_tensor: Optional[Tensor] = None,
        references_tensor: Optional[Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cur_metrics = {}

        if self.datasets_metrics:
            assert predictions is not None and references is not None
            for key in self.datasets_metrics:
                if key == "bleu":
                    self.datasets_metrics[key].add_batch(
                        predictions=[[token.lower() for token in line.split()] for line in predictions],
                        references=[[[token.lower() for token in line.split()]] for line in references],
                    )
                elif key == "chrf":
                    self.datasets_metrics[key].add_batch(
                        predictions=predictions, references=[[line] for line in references]
                    )
                else:
                    self.datasets_metrics[key].add_batch(predictions=predictions, references=references)
        if self.tensors_metrics:
            assert predictions_tensor is not None and references_tensor is not None
            cur_metrics = self.tensors_metrics(predictions_tensor, references_tensor)

        if self.strings_metrics:
            assert predictions is not None and references is not None
            cur_string_metrics = self.strings_metrics(predictions, references)
            if cur_metrics:
                cur_metrics.update(cur_string_metrics)
            else:
                cur_metrics = cur_string_metrics

        return cur_metrics

    def compute(self) -> Dict[str, float]:
        results = {}
        if self.datasets_metrics:
            for key in self.datasets_metrics:
                if key == "bleu":
                    results[key] = self.datasets_metrics[key].compute(smooth=True)["bleu"]
                elif key == "rouge":
                    rouge = self.datasets_metrics[key].compute()
                    results["rouge1"] = rouge["rouge1"].mid.fmeasure
                    results["rouge2"] = rouge["rouge2"].mid.fmeasure
                    results["rougeL"] = rouge["rougeL"].mid.fmeasure
                elif key == "meteor":
                    results[key] = self.datasets_metrics[key].compute()["meteor"]
                elif key == "b_norm":
                    results[key] = self.datasets_metrics[key].compute()["b_norm"]
                elif key == "chrf":
                    results[key] = self.datasets_metrics[key].compute()["score"] / 100

        for metrics in (self.tensors_metrics, self.strings_metrics):
            if metrics:
                metrics = metrics.compute()
                results.update({key: metrics[key] for key in metrics})

        if self.prefix:
            results = {f"{self.prefix}_{key}": results[key] for key in results}
        return results
