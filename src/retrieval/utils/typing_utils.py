import numpy.typing as npt
from typing_extensions import TypedDict


class CommitEmbeddingExample(TypedDict):
    diff_embedding: npt.NDArray
    pos_in_file: int


class RetrievalPrediction(TypedDict):
    pos_in_file: int
    distance: float
