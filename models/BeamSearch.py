import math
from typing import List, Tuple

import torch

from models.Search import Search


class BeamSearch(Search):
    """Beam search algorithm with normalized by length scores"""

    def __init__(self, eos_ids: List[int], vocab_size: int, beam_size: int, alpha=.0):
        super().__init__(eos_ids, vocab_size, beam_size)

        self._length = 1.0
        self._scores = None
        self._hypotheses = None
        self._terminated_hypotheses = []
        self._sort_mask = None
        self._eos_tensor = None
        self.alpha = alpha

    def _init_state(self, dtype: torch.dtype, device: torch.device):
        self._device = device
        self._scores = torch.zeros(1, dtype=dtype, device=device)
        self._hypotheses = torch.empty(1, 0, dtype=torch.long, device=device)
        self._eos_tensor = torch.tensor(self._eos_ids, dtype=torch.long, device=device).unsqueeze(1)

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
        super()._step_check(log_probs)
        if self._scores is None:
            assert self._hypotheses is None
            self._init_state(log_probs.dtype, log_probs.device)

        log_probs.add_(self._scores.unsqueeze(1))
        log_probs = log_probs.flatten()
        sample_scores, samples = torch.topk(
            log_probs,
            # Take more to ensure that we will keep search_size not terminated
            min((1 + len(self._eos_ids)) * self._search_size,
                (log_probs != -math.inf).sum().item() if possible_infs else log_probs.size(0))
        )

        sort_mask = torch.div(samples, self._vocab_size)
        samples.fmod_(self._vocab_size)

        self._init_sort_mask()
        self._update_state(samples, sample_scores, sort_mask)
        self._length += 1

        return self._sort_mask

    @property
    def hypotheses(self) -> List[List[Tuple[torch.Tensor, float]]]:
        """List of tuples of terminated hypotheses and theirs scores"""
        hypotheses = self._terminated_hypotheses
        return [sorted(hypotheses, key=lambda x: x[1], reverse=True)]

    @property
    def last_predictions(self) -> torch.Tensor:
        """Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model"""
        assert (
                self._hypotheses is not None and self._hypotheses.size(1) > 0
        ), f"Can't get last predictions if no steps have been performed"
        return self._hypotheses[:, -1]

    @property
    def batch_size(self) -> int:
        """Current batch size"""
        if self._scores is None:
            return 1
        return self._scores.size(0)

    def _init_sort_mask(self):
        self._sort_mask = torch.arange(self.batch_size)

    def _update_state(self, samples: torch.Tensor, sample_scores: torch.Tensor, sort_mask: torch.Tensor):
        self._sort_state(sort_mask)

        self._scores = sample_scores
        self._hypotheses = torch.cat((self._hypotheses, samples.unsqueeze(1)), dim=1)
        self._stash_terminated(samples)

    def _stash_terminated(self, samples: torch.Tensor):
        # We want to stash tokens only from the first search_size
        to_stash = self._is_sample_terminates(samples[:self._search_size])

        scores = self._scores / (self._length ** self.alpha)
        for terminated_hypothesis, score in zip(
                self._hypotheses[: self._search_size][to_stash], scores[: self._search_size][to_stash]
        ):
            assert len(terminated_hypothesis) == int(self._length)
            self._terminated_hypotheses.append((terminated_hypothesis.clone(), score.item()))

        # And throw out all terminated
        terminated = self._is_sample_terminates(samples)
        self._apply_slice_to_state(~terminated)
        self._sort_state()

    def _sort_state(self, sort_mask: torch.Tensor = None):
        if sort_mask is None:
            _, sort_mask = torch.topk(self._scores, min(self._search_size, self._scores.size(0)))
        self._apply_slice_to_state(sort_mask)

    def _is_sample_terminates(self, samples: torch.Tensor):
        result = samples == self._eos_tensor.expand(self._eos_tensor.size(0), samples.size(0))
        return result.sum(dim=0, dtype=torch.bool)

    def _apply_slice_to_state(self, tensor_slice):
        self._scores = self._scores[tensor_slice]
        self._hypotheses = self._hypotheses[tensor_slice]
        if self._sort_mask is not None:
            self._sort_mask = self._sort_mask[tensor_slice]