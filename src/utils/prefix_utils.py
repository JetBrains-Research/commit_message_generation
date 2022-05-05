from typing import Dict, List

import torch
from transformers import PreTrainedTokenizerBase


class PrefixAllowedTokens:
    def __init__(self, context_len: Dict[int, int], prefix: Dict[int, str], tokenizer: PreTrainedTokenizerBase):
        self._context_len = context_len
        self._prefix = prefix
        self._tokenizer = tokenizer

    def __call__(self, batch_id: int, sentence: torch.Tensor) -> List[int]:
        vocab = self._tokenizer.get_vocab()
        decoded_sentence = self._tokenizer.decode(sentence[self._context_len[batch_id] :])

        # when given prefix is empty, we can generate any token
        if not self._prefix:
            return list(vocab.values())

        # if we haven't generated prefix or its part yet, we can:
        # 1) generate tokens starting with the prefix
        # 2) generate tokens which are prefixes for the prefix
        if len(decoded_sentence) == 0:
            res = [
                vocab[key]
                for key in vocab
                if key.replace("Ġ", " ").startswith(self._prefix[batch_id])
                or self._prefix[batch_id].startswith(key.replace("Ġ", " "))
            ]
        # if we've already generated the prefix, we can generate any token
        elif decoded_sentence.startswith(self._prefix[batch_id]):
            res = list(vocab.values())
        # if we've generated only part of the prefix, we can:
        # 1) generate tokens starting with its remaining part
        # 2) generate tokens which are prefixes for its remaining part
        else:
            res = [
                vocab[key]
                for key in vocab
                if key.startswith(self._prefix[batch_id][len(decoded_sentence) :])
                or self._prefix[batch_id][len(decoded_sentence) :].startswith(key)
            ]

        return res
