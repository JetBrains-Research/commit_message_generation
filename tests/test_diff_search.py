import os
from typing import List

import numpy as np
import pytest

from src.retrieval import DiffSearch
from src.retrieval.utils import CommitEmbeddingExample, RetrievalPrediction


def cosine_sim(x: List[float], y: List[float]) -> float:
    """A simple helper function to compute cosine similarity between two 1D lists."""
    assert len(x) == len(y)
    xy = sum(x_item * y_item for x_item, y_item in zip(x, y))
    x_norm = sum(x_item**2 for x_item in x) ** 0.5
    y_norm = sum(y_item**2 for y_item in y) ** 0.5
    return xy / (x_norm * y_norm)


def angular_dist(x: List[float], y: List[float]) -> float:
    """A simple helper function to compute angular distance between two 1D lists."""
    assert len(x) == len(y)
    return (2 * (1 - cosine_sim(x, y))) ** 0.5


def test_idxs_out_of_order(tmp_path):
    search = DiffSearch(embeddings_dim=3, num_trees=3, metric="angular", load_index=False)
    search.add(CommitEmbeddingExample(diff_embedding=np.array([0, 0, 1]), pos_in_file=1))
    search.add(CommitEmbeddingExample(diff_embedding=np.array([1, 1, 1]), pos_in_file=2))
    search.add(CommitEmbeddingExample(diff_embedding=np.array([1, 0, 1]), pos_in_file=0))
    search.finalize()

    assert search._index.get_item_vector(0) == [1, 0, 1]
    assert search._index.get_item_vector(1) == [0, 0, 1]
    assert search._index.get_item_vector(2) == [1, 1, 1]


def test_train_logic(tmp_path):
    search = DiffSearch(embeddings_dim=3, num_trees=3, metric="angular", load_index=False)
    search.add(CommitEmbeddingExample(diff_embedding=np.array([1, 0, 1]), pos_in_file=0))
    search.add(CommitEmbeddingExample(diff_embedding=np.array([0, 0, 1]), pos_in_file=1))
    search.add(CommitEmbeddingExample(diff_embedding=np.array([1, 1, 1]), pos_in_file=2))
    search.finalize()

    # searching for the nearest neighbor for embedding present in index returns itself
    prediction = search.predict(np.array([1, 0, 1]), is_train=False)
    assert prediction == RetrievalPrediction(
        distance=pytest.approx(angular_dist([1, 0, 1], [1, 0, 1]), abs=1e-7),
        pos_in_file=0,
    )

    # but passing is_train=True fixes it
    prediction = search.predict(np.array([1, 0, 1]), is_train=True)
    assert prediction == RetrievalPrediction(
        distance=pytest.approx(angular_dist([1, 0, 1], [1, 1, 1]), abs=1e-7),
        pos_in_file=2,
    )


def test_nn_search(tmp_path):
    search = DiffSearch(embeddings_dim=3, num_trees=3, metric="angular", load_index=False)
    search.add(CommitEmbeddingExample(diff_embedding=np.array([0, 0, 1]), pos_in_file=1))
    search.add(CommitEmbeddingExample(diff_embedding=np.array([1, 1, 1]), pos_in_file=2))
    search.add(CommitEmbeddingExample(diff_embedding=np.array([1, 0, 1]), pos_in_file=0))
    search.finalize()

    prediction = search.predict(np.array([1, 0, 1]), is_train=False)
    assert prediction == RetrievalPrediction(
        distance=pytest.approx(angular_dist([1, 0, 1], [1, 0, 1]), abs=1e-7),
        pos_in_file=0,
    )
