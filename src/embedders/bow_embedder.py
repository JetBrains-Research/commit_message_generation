import json
from typing import Iterator, List, Mapping, Optional

import numpy.typing as npt
from sklearn.feature_extraction.text import CountVectorizer

from src.embedders.base_embedder import BaseEmbedder


class BagOfWordsEmbedder(BaseEmbedder):
    """
    This class is used to construct simple bag-of-words embeddings.
    """

    def __init__(self, vocabulary: Optional[Mapping[str, int]] = None, max_features: Optional[int] = None) -> None:
        self._vectorizer = CountVectorizer(vocabulary=vocabulary, max_features=max_features)

    @property
    def vocab(self):
        try:
            return self._vectorizer.vocabulary_
        except AttributeError:
            return {}

    def save_vocab(self, vocab_filename: str):
        with open(vocab_filename, "w") as f:
            json.dump({key: int(self.vocab[key]) for key in self.vocab}, f)

    def fit(self, input_content: Iterator[str], *args, **kwargs) -> None:
        self._vectorizer.fit(input_content)

    def transform(self, input_content: List[str], *args, **kwargs) -> npt.NDArray:
        return self._vectorizer.transform(input_content).toarray()

    @property
    def embeddings_dim(self):
        return len(self.vocab)
