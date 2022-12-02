import json
from typing import List, Mapping, Optional

import numpy.typing as npt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

from src.embedders.base_embedder import BaseEmbedder
from src.utils import mods_to_diff


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

    def build_vocab(self, input_filename: str, chunksize: int) -> None:
        """Given input file, builds vocabulary by iterating over it in chunks."""
        reader = pd.read_json(input_filename, orient="records", lines=True, chunksize=chunksize)
        it = (
            [mods_to_diff(mods) for mods in chunk["mods"].tolist()]
            for chunk in tqdm(reader, desc=f"Iterating over {input_filename} to fit embedder")
        )
        self._vectorizer.fit(item for lst in it for item in lst)

    def _transform(self, diffs: List[str], *args, **kwargs) -> npt.NDArray:
        return self._vectorizer.transform(diffs).toarray()

    @property
    def embeddings_dim(self):
        return len(self.vocab)
