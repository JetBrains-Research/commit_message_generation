import os
from typing import List

import jsonlines
import numpy as np
import pytest

from src.search import DiffSearch
from src.search.diff import RetrievalPrediction


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
    os.chdir(tmp_path)
    search = DiffSearch(num_neighbors=1, embeddings_dim=3, input_fname="", num_trees=3)
    search.add(np.array([0, 0, 1]), idx=1)
    search.add(np.array([1, 1, 1]), idx=2)
    search.add(np.array([1, 0, 1]), idx=0)
    search.finalize()

    assert search._index.get_item_vector(0) == [1, 0, 1]
    assert search._index.get_item_vector(1) == [0, 0, 1]
    assert search._index.get_item_vector(2) == [1, 1, 1]


def test_nn_search(tmp_path):
    os.chdir(tmp_path)
    input_fname = "input.jsonl"
    with jsonlines.open(input_fname, "w") as writer:
        writer.write_all(
            [
                {
                    "message": f"message {i}",
                    "mods": [{"change_type": "MODIFY", "old_path": "path", "new_path": "path", "diff": f"diff {i}"}],
                }
                for i in range(3)
            ]
        )

    search = DiffSearch(num_neighbors=1, embeddings_dim=3, input_fname=str(input_fname), num_trees=3)
    search.add(np.array([0, 0, 1]), idx=1)
    search.add(np.array([1, 1, 1]), idx=2)
    search.add(np.array([1, 0, 1]), idx=0)
    search.finalize()

    prediction = search.predict(np.array([1, 0, 1]), "path[NL]diff 2[NL]")
    assert prediction == RetrievalPrediction(
        message="message 0",
        diff="path[NL]diff 0[NL]",
        distance=pytest.approx(angular_dist([1, 0, 1], [1, 0, 1]), abs=1e-7),
    )


def test_2nn_search(tmp_path):
    os.chdir(tmp_path)
    input_fname = "input.jsonl"
    with jsonlines.open(input_fname, "w") as writer:
        writer.write_all(
            [
                {
                    "message": f"message {i}",
                    "mods": [{"change_type": "MODIFY", "old_path": "path", "new_path": "path", "diff": f"diff {i}"}],
                }
                for i in range(3)
            ]
        )

    search = DiffSearch(num_neighbors=2, embeddings_dim=3, input_fname=str(input_fname), num_trees=3)

    search.add(np.array([0, 0, 1]), idx=1)
    search.add(np.array([1, 1, 1]), idx=2)
    search.add(np.array([1, 0, 1]), idx=0)
    search.finalize()

    # 2 nearest neighbors of [1, 0, 1] from this index: [1, 0, 1] (#0) and [1, 1, 1] (#2)
    # we calculate BLEU between given diff and diffs of nearest neighbors
    # BLEU("diff 2", "diff 2") > BLEU("diff 1", "diff 2") => we return message from example #2
    prediction = search.predict(np.array([1, 0, 1]), "path[NL]diff 2[NL]")
    assert prediction == RetrievalPrediction(
        message="message 2",
        diff="path[NL]diff 2[NL]",
        distance=pytest.approx(angular_dist([1, 0, 1], [1, 1, 1])),
    )
