from typing import List, Optional

import torch
from torchmetrics import Metric


class ExactMatch(Metric):
    """
    ExactMatch@N metric. Given N, calculates the ratio of examples
    where first N generated words exactly match first N words from corresponding target.

    Words are obtained by splitting by whitespaces. Cases where target contains less than N words are skipped.

    Args:
        n: Number of words to compare. Optional, full sequences will be considered if n is not given.
    """

    def __init__(self, n: Optional[int] = None, dist_sync_on_step: bool = False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n = n

        self.correct: torch.Tensor
        self.total: torch.Tensor
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: List[str], references: List[str]) -> None:  # type: ignore[override]
        for pred, ref in zip(predictions, references):
            pred_words, ref_words = pred.strip().split(), ref.strip().split()

            if self.n:
                # compare first n words
                if len(ref_words) >= self.n:
                    if len(pred_words) >= self.n and all(
                        pred_word == target_word
                        for pred_word, target_word in zip(pred_words[: self.n], ref_words[: self.n])
                    ):
                        self.correct += 1

                    self.total += 1
            else:
                # compare full sequences
                if len(pred_words) == len(ref_words) and all(
                    pred_word == target_word for pred_word, target_word in zip(pred_words, ref_words)
                ):
                    self.correct += 1
                self.total += 1

    def compute(self) -> torch.Tensor:
        if not self.total.item():
            return self.total

        return self.correct.float() / self.total
