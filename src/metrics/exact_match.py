import torch
import re
from typing import List
from string import punctuation
from torchmetrics import Metric


class ExactMatch(Metric):
    def __init__(self, n: int, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n = n

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: List[str], target: List[str]):
        for pred_example, target_example in zip(pred, target):
            for pred_word, target_word in zip(
                re.split("[ \n\t\r" + punctuation + "]+", pred_example)[: self.n],
                re.split("[ \n\t\r" + punctuation + "]+", target_example)[: self.n],
            ):
                if pred_word == target_word:
                    self.correct += 1

            self.total += self.n

    def compute(self):
        return self.correct.float() / self.total
