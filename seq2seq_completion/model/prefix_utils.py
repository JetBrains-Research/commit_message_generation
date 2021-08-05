import torch
from typing import List
from transformers import PreTrainedTokenizerBase


class PrefixAllowedTokens:
    def __init__(self, context_len: int, prefix: str, tokenizer: PreTrainedTokenizerBase, num_beams: int):
        self._context_len = context_len
        self._prefix = prefix
        self._tokenizer = tokenizer
        self._num_beams = num_beams

    def __call__(self, _: int, sentence: torch.Tensor) -> List[int]:
        vocab = self._tokenizer.get_vocab()
        decoded_sentence = self._tokenizer.decode(sentence[self._context_len :])

        # if we haven't generated prefix or its part yet, we can:
        # 1
        if len(decoded_sentence) == 0:
            res = [
                vocab[key]
                for key in vocab
                if key.replace("Ġ", " ").startswith(self._prefix) or self._prefix.startswith(key.replace("Ġ", " "))
            ]
        # if we've already generated the prefix, we can generate any token
        elif decoded_sentence.startswith(self._prefix):
            res = list(vocab.values())
        # if we've generated only part of the prefix, we can:
        # 1) generate tokens starting with its remaining part
        else:
            res = [
                vocab[key]
                for key in vocab
                if key.startswith(self._prefix[len(decoded_sentence) :])
                or self._prefix[len(decoded_sentence) :].startswith(key)
            ]

        return res
