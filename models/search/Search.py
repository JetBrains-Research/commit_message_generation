from typing import List, Tuple

import torch


class Search(object):
    """
    Class for search algorithms
    Basically user needs to feed log_probs and perform a step several times
    Results can be found in hypotheses
    """

    def __init__(self, eos_ids: List[int], vocab_size: int, search_size: int):
        self._eos_ids = eos_ids
        self._search_size = search_size
        self._vocab_size = vocab_size

    def step(self, log_probs: torch.Tensor, possible_infs: bool = False) -> torch.Tensor:
        """Take a single search step.
        Args:
            log_probs: (batch_size, vocab_size)
                the model's log-probabilities over the vocabulary at the current step
            possible_infs: whether log_probs can contain -inf
        Return:
            beams: (batch_size,)
                the hypothesis ids of the chosen elements, in the range [0, batch_size)
        """
        raise NotImplementedError

    def _step_check(self, log_probs: torch.Tensor):
        assert log_probs.size() == (
            self.batch_size,
            self._vocab_size,
        ), f"log_probs must have shape {(self.batch_size, self._vocab_size)}, but {log_probs.size()} was given"

        assert all(
            eos < self._vocab_size for eos in self._eos_ids
        ), f"EOS ids must be less than vocab_size, but EOS ids: {self._eos_ids} and vocab_size: {self._vocab_size}"

    @property
    def hypotheses(self) -> List[List[Tuple[torch.Tensor, float]]]:
        """List of list of tuples of terminated hypotheses and theirs scores"""
        raise NotImplementedError

    @property
    def last_predictions(self) -> torch.Tensor:
        """Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model"""
        raise NotImplementedError

    @property
    def batch_size(self) -> int:
        """Current batch size"""
        raise NotImplementedError