import os
from typing import List, Literal

import faiss
import numpy.typing as npt
import numpy as np
from ..utils import CommitEmbeddingExample, RetrievalPrediction


class FaissSearch:
    """This class is used to retrieve the nearest neighbor via the Annoy library."""

    def __init__(self, embeddings_dim: int, load_index: bool, index_root_dir: str = ".", device: str = "cpu") -> None:

        self._embeddings_dim = embeddings_dim
        self._index_root_dir = index_root_dir

        self.__index = faiss.IndexFlatIP(embeddings_dim)
        self._index = faiss.IndexIDMap(self.__index)

        if load_index:
            if "saved.index" not in os.listdir(index_root_dir):
                raise ValueError("Configured to load pretrained index, but it doesn't exist!")
            else:
                self._index = faiss.read_index("saved.index", faiss.METRIC_INNER_PRODUCT)

        if device == "cuda":
            self._index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self._index)

    def add(self, example: CommitEmbeddingExample) -> None:
        """Adds a single item to the index."""
        self._index.add_with_ids(example["diff_embedding"], example["pos_in_file"])

    def add_batch(self, batch: List[CommitEmbeddingExample]) -> None:
        """Adds a batch of items to the index."""
        embeddings: npt.NDArray = np.asarray([example["diff_embedding"] for example in batch])
        ids: npt.NDArray = np.asarray([example["pos_in_file"] for example in batch])
        assert embeddings.shape == (len(batch), self._embeddings_dim)

        self._index.add_with_ids(embeddings, ids)

    def finalize(self) -> None:
        faiss.write_index(self._index, "saved.index")

    def predict(self, diff_embedding: npt.NDArray, is_train: bool) -> RetrievalPrediction:
        """Retrieves the closest neighbor for given diff from index."""
        num_neighbors = 2 if is_train else 1

        retrieved_distances, retrieved_idxs = self._index.search(diff_embedding, num_neighbors)

        if is_train:
            retrieved_idxs, retrieved_distances = retrieved_idxs[1:], retrieved_distances[1:]

        return RetrievalPrediction(
            distance=float(retrieved_distances[0]),
            pos_in_file=retrieved_idxs[0],
        )

    def predict_batch(self, batch: List[CommitEmbeddingExample], is_train: bool) -> List[RetrievalPrediction]:
        """Retrieves the closest neighbors for each example in a batch."""
        embeddings: npt.NDArray = np.asarray([example["diff_embedding"] for example in batch])
        assert embeddings.shape == (len(batch), self._embeddings_dim)

        num_neighbors = 2 if is_train else 1
        retrieved_distances, retrieved_idxs = self._index.search(embeddings, num_neighbors)

        if is_train:
            retrieved_idxs, retrieved_distances = retrieved_idxs[:, 1:], retrieved_distances[:, 1:]

        return [
            RetrievalPrediction(
                distance=distance,
                pos_in_file=idx,
            )
            for distance, idx in zip(retrieved_distances, retrieved_idxs)
        ]
