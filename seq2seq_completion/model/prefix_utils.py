import torch
import math
from typing import List
from transformers import PreTrainedTokenizerBase, LogitsProcessor


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, prefix: str, tokenizer: PreTrainedTokenizerBase, num_beams: int):
        super(PrefixConstrainedLogitsProcessor, self).__init__()
        self._prefix = prefix
        self._tokenizer = tokenizer
        self._num_beams = num_beams
        self.already_generated = {i: False for i in range(self._num_beams)}

    def _prefix_allowed_tokens_fn(self, beam_id: int, sentence: torch.Tensor) -> List[int]:
        vocab = self._tokenizer.get_vocab()
        decoded_sentence = self._tokenizer.decode(sentence)

        # if we've already generated the prefix, we can generate any token
        if self.already_generated[beam_id]:
            res = list(vocab.values())

        # if we've generated only part of the prefix, we can:
        elif self._prefix.startswith(" " + decoded_sentence.split()[-1]):
            generated_part = " " + decoded_sentence.split()[-1]
            # 1) generate tokens starting with its remaining part
            res = [vocab[key] for key in vocab if key.startswith(self._prefix[len(generated_part) :])]
            if len(res) > 0:
                self.already_generated[beam_id] = True
            else:
                # 2) generate tokens which are prefixes for its remaining part (which is less preferred)
                res = [vocab[key] for key in vocab if self._prefix[len(generated_part) :].startswith(key)]

        # if we haven't generated prefix or its part yet, we can:
        else:
            # 1) generate tokens starting with the prefix
            res = [vocab[key] for key in vocab if key.replace("Ġ", " ").startswith(self._prefix)]
            if len(res) > 0:
                self.already_generated[beam_id] = True
            else:
                # 2) generate tokens which are prefixes for the prefix (which is less preferred)
                res = [vocab[key] for key in vocab if self._prefix.startswith(key.replace("Ġ", " "))]

        return res

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.Tensor:
        """
        For each beam hypothesis:
        - get list of allowed tokens (based on current hypothesis and given prefix)
        - make sure that only allowed tokens will be generated by setting all other tokens' probabilities to -inf
        """
        mask = torch.full_like(scores, -math.inf)
        for beam_id, sent in enumerate(input_ids):
            mask[beam_id, self._prefix_allowed_tokens_fn(beam_id, sent)] = 0.0
        return scores + mask
