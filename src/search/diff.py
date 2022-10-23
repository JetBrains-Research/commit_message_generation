import json
from linecache import getline
from typing import List

import annoy
import numpy as np
import numpy.typing as npt

from src.metrics import b_norm_score
from src.utils import RetrievalExample, RetrievalPrediction, mods_to_diff


class DiffSearch:
    """This class is used to retrieve nearest neighbors with the help of Annoy library."""

    def __init__(self, num_neighbors: int, num_trees: int, embeddings_dim: int, input_fname: str) -> None:
        self._num_neighbors = num_neighbors
        self._num_trees = num_trees
        self._index = annoy.AnnoyIndex(embeddings_dim, "angular")
        self._index.set_seed(42)
        self._index.on_disk_build(f"index_{num_trees}.ann")
        self._input_fname = input_fname

    def add(self, diff_input_ids: npt.NDArray, idx: int) -> None:
        self._index.add_item(idx, diff_input_ids)

    def add_batch(self, batch: List[RetrievalExample]) -> None:
        for example in batch:
            if len(example.diff_input_ids.shape) > 1:
                assert example.diff_input_ids.shape[0] == 1
                example.diff_input_ids = example.diff_input_ids.flatten()
            self.add(diff_input_ids=example.diff_input_ids, idx=example.idx)

    def finalize(self) -> None:
        self._index.build(self._num_trees)

    def load_index(self, index_fname: str):
        self._index.load(index_fname)

    def _find_nn(self, retrieved_diffs: List[str], test_diff: str) -> int:
        scores = [
            b_norm_score(predictions=[test_diff], references=[retrieved_diff]) for retrieved_diff in retrieved_diffs
        ]
        return int(np.argmax(np.array(scores)))

    def predict(self, diff_input_ids: npt.NDArray, diff: str) -> RetrievalPrediction:
        if len(diff_input_ids.shape) > 1:
            assert diff_input_ids.shape[0] == 1
            diff_input_ids = diff_input_ids.flatten()

        retrieved_idxs, retrieved_distances = self._index.get_nns_by_vector(
            diff_input_ids, self._num_neighbors, include_distances=True
        )
        retrieved_data = [json.loads(getline(self._input_fname, idx + 1).rstrip("\n")) for idx in retrieved_idxs]

        nn_idx = self._find_nn([mods_to_diff(data["mods"]) for data in retrieved_data], diff)
        return RetrievalPrediction(
            message=retrieved_data[nn_idx]["message"],
            diff=mods_to_diff(retrieved_data[nn_idx]["mods"]),
            distance=float(retrieved_distances[nn_idx]),
        )

    def predict_batch(self, batch: List[RetrievalExample]) -> List[RetrievalPrediction]:
        predictions = []
        for example in batch:
            if len(example.diff_input_ids.shape) > 1:
                assert example.diff_input_ids.shape[0] == 1
                example.diff_input_ids = example.diff_input_ids.flatten()
            predictions.append(self.predict(diff_input_ids=example.diff_input_ids, diff=example.diff))
        return predictions
