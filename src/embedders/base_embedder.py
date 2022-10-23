from typing import List

import pandas as pd
from tqdm import tqdm

from src.utils import mods_to_diff


class BaseEmbedder:
    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def fit_full_file(self, input_filename: str, chunksize: int):
        reader = pd.read_json(input_filename, orient="records", lines=True, chunksize=chunksize)
        it = (
            [mods_to_diff(mods) for mods in chunk["mods"].tolist()]
            for chunk in tqdm(reader, desc=f"Iterating over {input_filename} to fit embedder")
        )
        self.fit(item for lst in it for item in lst)

    def transform(self, text: List[str], *args, **kwargs) -> List[int]:
        raise NotImplementedError

    @property
    def embeddings_dim(self):
        raise NotImplementedError
