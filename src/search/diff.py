import json
from linecache import getline
from typing import List, Literal

import annoy
import numpy as np
import numpy.typing as npt

from src.metrics import b_norm_score
from src.utils import CommitEmbeddingExample, RetrievalPrediction, mods_to_diff


class DiffSearch:
    """This class is used to retrieve nearest neighbors with the help of Annoy library."""

    def __init__(
        self,
        num_neighbors: int,
        num_trees: int,
        embeddings_dim: int,
        input_fname: str,
        metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"] = "angular",
    ) -> None:
        self._num_neighbors = num_neighbors
        self._num_trees = num_trees
        self._index = annoy.AnnoyIndex(embeddings_dim, metric)
        self._index.set_seed(42)
        self._index.on_disk_build(f"index_{num_trees}.ann")
        self._input_fname = input_fname

    def add(self, diff_input_ids: npt.NDArray, idx: int) -> None:
        """Add a single item to index."""
        self._index.add_item(idx, diff_input_ids)

    def add_batch(self, batch: List[CommitEmbeddingExample]) -> None:
        """Add a batch of items to index.

        Simply iterates over batch, because annoy doesn't support batches processing.
        """
        for example in batch:
            if len(example["diff_embedding"].shape) > 1:
                assert example["diff_embedding"].shape[0] == 1
                example["diff_embedding"] = example["diff_embedding"].flatten()
            self.add(diff_input_ids=example["diff_embedding"], idx=example["idx"])

    def finalize(self) -> None:
        self._index.build(self._num_trees)

    def load_index(self, index_fname: str):
        """Loads prebuilt index from given file."""
        self._index.load(index_fname)

    def _find_nn(self, retrieved_diffs: List[str], test_diff: str) -> int:
        """Given the list of "num_neighbors" retrieved results, returns the idx of the closest neighbor.

        * When num_neighbors = 1, does nothing and returns 0.

        * When num_neighbors > 1, approach from NNGen paper is used: returns the ids of retrieved diff with the highest
          BLEU score with target diff.
        """
        if self._num_neighbors == 1:
            return 0

        scores = [
            b_norm_score(predictions=[test_diff], references=[retrieved_diff]) for retrieved_diff in retrieved_diffs
        ]
        return int(np.argmax(np.array(scores)))

    def predict(self, diff_embedding: npt.NDArray, diff: str) -> RetrievalPrediction:
        """Retrieves the closest neighbor for given diff from index.

        When "num_neighbors" > 1, retrieves "num_neighbors" closest examples and then utilizes approach from NNGen paper
        to choose one.
        """
        if len(diff_embedding.shape) > 1:
            assert diff_embedding.shape[0] == 1
            diff_embedding = diff_embedding.flatten()

        retrieved_idxs, retrieved_distances = self._index.get_nns_by_vector(
            diff_embedding, self._num_neighbors, include_distances=True
        )
        retrieved_data = [json.loads(getline(self._input_fname, idx + 1).rstrip("\n")) for idx in retrieved_idxs]

        nn_idx = self._find_nn([mods_to_diff(data["mods"]) for data in retrieved_data], diff)
        return RetrievalPrediction(
            message=retrieved_data[nn_idx]["message"],
            diff=mods_to_diff(retrieved_data[nn_idx]["mods"]),
            distance=float(retrieved_distances[nn_idx]),
            idx=retrieved_idxs[nn_idx],
        )

    def predict_batch(self, batch: List[CommitEmbeddingExample]) -> List[RetrievalPrediction]:
        """Retrieve closest neighbors for a batch of items.

        Simply iterates over batch, because annoy doesn't support batches processing.
        """
        predictions = []
        for example in batch:
            if len(example["diff_embedding"].shape) > 1:
                assert example["diff_embedding"].shape[0] == 1
                example["diff_embedding"] = example["diff_embedding"].flatten()
            predictions.append(self.predict(diff_embedding=example["diff_embedding"], diff=example["diff"]))
        return predictions
