from typing import List, Optional

import torch
from rapidfuzz.distance.Levenshtein import normalized_similarity
from torchmetrics import Metric


class EditSimilarity(Metric):
    """Edit Similarity metric. It is a string similarity metric based on Levenshtein distance:
     1 - edit_distance/max_len

    Final metric value is calculated as average sentence-level edit similarity.
    """

    def __init__(
        self,
        insertion_cost: int = 1,
        deletion_cost: int = 1,
        substitution_cost: int = 1,
        dist_sync_on_step: bool = False,
    ) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.weights = (insertion_cost, deletion_cost, substitution_cost)

        self.scores: torch.Tensor
        self.total: torch.Tensor
        self.add_state("scores", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: List[str], references: List[str]) -> None:  # type: ignore[override]
        for pred, ref in zip(predictions, references):
            e_sim = normalized_similarity(
                pred,
                ref,
                weights=self.weights,
            )

            if not ref:
                self.scores = torch.tensor(float("nan"))
            else:
                self.scores += torch.tensor(e_sim)
            self.total += 1

    def compute(self) -> torch.Tensor:
        return self.scores / self.total
