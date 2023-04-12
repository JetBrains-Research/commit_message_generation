import os
from typing import List, Literal

import annoy
import numpy.typing as npt

from ..utils import CommitEmbeddingExample, RetrievalPrediction


class DiffSearch:
    """This class is used to retrieve the nearest neighbor via the Annoy library."""

    def __init__(
        self,
        num_trees: int,
        embeddings_dim: int,
        load_index: bool,
        load_index_path: str = ".",
        index_root_dir: str = ".",
        metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"] = "angular",
    ) -> None:
        self._num_trees = num_trees

        self._index = annoy.AnnoyIndex(embeddings_dim, metric)
        self._index.set_seed(42)

        if load_index:
            if f"index_{num_trees}.ann" not in os.listdir(index_root_dir):
                raise ValueError("Configured to load pretrained index, but it doesn't exist!")
            else:
                self._index.load(load_index_path)
        else:
            self._index.on_disk_build(os.path.join(index_root_dir, f"index_{num_trees}.ann"))

    def add(self, example: CommitEmbeddingExample) -> None:
        """Adds a single item to the index."""
        self._index.add_item(example["pos_in_file"], example["diff_embedding"])

    def add_batch(self, batch: List[CommitEmbeddingExample]) -> None:
        """Adds a batch of items to the index.

        Note: Simply iterates over batch, because annoy doesn't support batch processing.
        """
        for example in batch:
            if len(example["diff_embedding"].shape) > 1:
                assert example["diff_embedding"].shape[0] == 1
                example["diff_embedding"] = example["diff_embedding"].flatten()
            self.add(example)

    def finalize(self) -> None:
        self._index.build(self._num_trees)

    def predict_train(self, idx: int) -> RetrievalPrediction:
        """Retrieves the closest neighbor for given idx of embedding already present in index."""
        # we are interested in the nearest neighbor, but for vectors from index it will always be themselves
        # so, we search for 2 neighbors and skip the first one
        retrieved_idxs, retrieved_distances = self._index.get_nns_by_item(idx, 2, include_distances=True)
        retrieved_idxs, retrieved_distances = retrieved_idxs[1:], retrieved_distances[1:]
        return RetrievalPrediction(
            distance=float(retrieved_distances[0]),
            pos_in_file=retrieved_idxs[0],
        )

    def predict(self, diff_embedding: npt.NDArray) -> RetrievalPrediction:
        """Retrieves the closest neighbor from index for given embedding."""

        if len(diff_embedding.shape) > 1:
            assert (
                diff_embedding.shape[0] == 1
            ), "This method is used to process single example. Use `predict_batch` to process several examples."
            diff_embedding = diff_embedding.flatten()

        retrieved_idxs, retrieved_distances = self._index.get_nns_by_vector(diff_embedding, 1, include_distances=True)

        return RetrievalPrediction(
            distance=float(retrieved_distances[0]),
            pos_in_file=retrieved_idxs[0],
        )

    def predict_batch(self, batch: List[CommitEmbeddingExample]) -> List[RetrievalPrediction]:
        """Retrieves the closest neighbors for each example in a batch.

        Note: Simply iterates over batch, because annoy doesn't support batch processing.
        """
        return [self.predict(diff_embedding=example["diff_embedding"]) for example in batch]

    def predict_batch_train(self, batch_idxs: List[int]) -> List[RetrievalPrediction]:
        """Retrieves the closest neighbors for each example in a batch. Intended for examples present in index.

        Note: Simply iterates over batch, because annoy doesn't support batch processing.
        """
        return [self.predict_train(idx=example) for example in batch_idxs]
