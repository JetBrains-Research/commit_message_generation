from abc import ABC, abstractmethod
from typing import List

import numpy.typing as npt

from src.utils import CommitEmbeddingExample, CommitTextExample


class BaseEmbedder(ABC):
    """Base class for embedders.

    Method "_transform" should do real work, "transform" uses its result to support correct input/output format.
    """

    @abstractmethod
    def _transform(self, diffs: List[str], *args, **kwargs) -> npt.NDArray:
        raise NotImplementedError

    def transform(self, inputs: List[CommitTextExample], *args, **kwargs) -> List[CommitEmbeddingExample]:
        diffs = [ex["diff"] for ex in inputs]
        diffs_embeddings: npt.NDArray = self._transform(diffs, *args, **kwargs)
        return [
            CommitEmbeddingExample(
                diff=input["diff"], message=input["message"], diff_embedding=embedding, idx=input["idx"]
            )
            for input, embedding in zip(inputs, diffs_embeddings)
        ]

    @property
    @abstractmethod
    def embeddings_dim(self):
        raise NotImplementedError
