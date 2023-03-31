from typing import Dict, List, Optional

import marisa_trie
import torch
from transformers import PreTrainedTokenizerFast


class VocabPrefixTree:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self._vocab = tokenizer.get_vocab()  # type: ignore[attr-defined]
        self._trie = marisa_trie.Trie([key.replace("Ġ", " ") for key in self._vocab])

    def get_tokens(self, prefix: str) -> List[int]:
        """Uses the trie to find all the tokens that either:

        * start with the given prefix
        * are prefixes for the given prefix

        Args:
            prefix: Current prefix.

        Returns:
            A list of tokens ids.
        """
        tokens = []
        for token in self._trie.prefixes(prefix):
            try:
                tokens.append(self._vocab[token])
            except KeyError:
                tokens.append(self._vocab[token.replace(" ", "Ġ")])

        for token in self._trie.keys(prefix):
            try:
                tokens.append(self._vocab[token])
            except KeyError:
                tokens.append(self._vocab[token.replace(" ", "Ġ")])

        return tokens


class PrefixAllowedTokens:
    def __init__(
        self,
        context_len: Dict[int, int],
        prefix: Dict[int, str],
        tokenizer: PreTrainedTokenizerFast,
        trie: Optional[VocabPrefixTree] = None,
    ):
        self._context_len = context_len
        self._prefix = prefix

        self._tokenizer = tokenizer
        if not trie:
            trie = VocabPrefixTree(tokenizer)
        self._trie = trie

    def __call__(self, batch_id: int, sentence: torch.Tensor) -> List[int]:
        decoded_sentence = self._tokenizer.decode(sentence[self._context_len[batch_id] :])  # type: ignore[attr-defined]

        # when given prefix is empty, we can generate any token
        if not self._prefix[batch_id]:
            return list(self._tokenizer.vocab.values())  # type: ignore[attr-defined]

        # if we haven't generated prefix or its part yet, we can:
        # 1) generate tokens starting with the prefix
        # 2) generate tokens which are prefixes for the prefix
        if len(decoded_sentence) == 0:
            res = self._trie.get_tokens(self._prefix[batch_id])
        # if we've already generated the prefix, we can generate any token
        elif decoded_sentence.startswith(self._prefix[batch_id]):
            res = list(self._tokenizer.vocab.values())  # type: ignore[attr-defined]
        # if we've generated only part of the prefix, we can:
        # 1) generate tokens starting with its remaining part
        # 2) generate tokens which are prefixes for its remaining part
        else:
            res = self._trie.get_tokens(self._prefix[batch_id][len(decoded_sentence) :])

        return res
