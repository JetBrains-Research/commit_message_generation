import torch
from typing import List, Optional
from torchmetrics import Metric
from rapidfuzz.distance.Levenshtein import normalized_similarity


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
        dist_sync_on_step: Optional[bool] = False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.weights = (insertion_cost, deletion_cost, substitution_cost)

        self.add_state("scores", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: List[str], references: List[str]) -> None:
        for pred, ref in zip(predictions, references):
            e_sim = normalized_similarity(
                pred,
                ref,
                weights=self.weights,
            )
            self.scores += torch.tensor(e_sim / 100.0)  # normalizing 1 - 100 score to 0.0 - 1.0 range
            self.total += 1

    def compute(self) -> float:
        return self.scores / self.total
