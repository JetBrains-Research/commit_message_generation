import torch
from typing import List, Optional
from torchmetrics import Metric
from src.metrics.reused_implementations import log_mnext_score


class LogMNEXT(Metric):
    """Log-MNEXT metric. It is a string similarity metric based on METEOR-NEXT.

    It was proposed in the "Evaluating Commit Message Generation: To BLEU Or Not To BLEU?" paper
    accepted to ICSE NIER 2022. This class uses original implementation from replication package.

    Final metric value is calculated as average sentence-level Log-MNEXT (replication package includes only
    sentence-level implementation).
    """

    def __init__(self, dist_sync_on_step: Optional[bool] = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("scores", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: List[str], references: List[str]) -> None:
        for pred, ref in zip(predictions, references):
            self.scores += torch.tensor(log_mnext_score([ref], pred))
            self.total += 1

    def compute(self) -> float:
        return self.scores / self.total
