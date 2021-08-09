import torch
from nltk import edit_distance
from typing import List
from torchmetrics import Metric


class EditSimilarity(Metric):
    def __init__(self, substitution_cost: int = 1, transpositions: bool = False, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.substitution_cost = substitution_cost
        self.transpositions = transpositions

        self.add_state("e_dist", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: List[str], target: List[str]):
        for pred_example, target_example in zip(pred, target):
            e_dist = edit_distance(
                pred_example,
                target_example,
                substitution_cost=self.substitution_cost,
                transpositions=self.transpositions,
            )
            self.e_dist += torch.tensor(e_dist)
            self.total += torch.tensor(max((len(pred_example), len(target_example))))

    def compute(self):
        return 1.0 - self.e_dist.float() / self.total
