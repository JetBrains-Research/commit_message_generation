from typing import List, Tuple

import torch

from models.search.BeamSearch import BeamSearch
from models.search.Search import Search


class DiverseBeamSearch(Search):
    """Beam search with diverse Hamming reward"""

    def __init__(
            self,
            eos_id: int,
            vocab_size: int,
            search_size: int,
            num_groups: int,
            diversity_strength: float,
    ):
        super().__init__(eos_id, vocab_size, search_size)

        self._num_groups = num_groups
        self._diversity_strength = -diversity_strength
        self._diversity_reward = None

        self._searches = [BeamSearch(eos_id, vocab_size, search_size) for _ in range(num_groups)]

    def _init_diversity_reward(self, dtype: torch.dtype, device: torch.device):
        if self._diversity_reward is None:
            self._diversity_reward = torch.zeros(1, self._vocab_size, dtype=dtype, device=device)
        else:
            self._diversity_reward[:] = 0.0

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
        self._init_diversity_reward(log_probs.dtype, log_probs.device)

        offset = 0
        beams_sort = []
        for beam in self._searches:
            old_batch_size = beam.batch_size

            cur_log_probs = log_probs[offset: offset + old_batch_size]
            cur_beams_sort = beam.step(cur_log_probs, possible_infs)
            beams_sort.append(cur_beams_sort + offset)

            # update diversity penalty
            self._diversity_reward.scatter_add_(
                1, beam.last_predictions.unsqueeze(0), self._diversity_reward.new_ones(1, beam.batch_size)
            )
            log_probs = torch.add(log_probs, self._diversity_strength, self._diversity_reward)

            offset += old_batch_size

        return torch.cat(beams_sort)

    @property
    def hypotheses(self) -> List[List[Tuple[torch.Tensor, float]]]:
        """List of groups of hypotheses, where group is a list of tuples of terminated hypotheses and theirs scores"""
        return [beam.hypotheses[0] for beam in self._searches]

    @property
    def last_predictions(self) -> torch.Tensor:
        """Tensor of last tokens of the current hypotheses with shape (batch_size,) to make a batch for a model"""
        return torch.cat([beam.last_predictions for beam in self._searches])

    @property
    def batch_size(self) -> int:
        """Current batch size"""
        return sum(beam.batch_size for beam in self._searches)