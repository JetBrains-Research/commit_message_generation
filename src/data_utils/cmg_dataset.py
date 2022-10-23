import json
from dataclasses import dataclass
from typing import Dict, Iterator, List

import jsonlines
import numpy as np
from torch.utils.data import IterableDataset, get_worker_info

from src.embedders import BaseEmbedder
from src.utils import RetrievalExample, mods_to_diff


class RetrievalDataset(IterableDataset):
    """Defines a dataset for retrieval approach to commit message generation task as IterableDataset"""

    def __init__(self, data_filename: str, embedder: BaseEmbedder):
        self._data_filename = data_filename
        self._embedder = embedder

    def __iter__(self) -> Iterator[RetrievalExample]:
        with jsonlines.open(self._data_filename) as reader:

            for i, line_content in enumerate(reader):
                if i % get_worker_info().num_workers == get_worker_info().id:
                    yield RetrievalExample(
                        diff_input_ids=self._embedder.transform([mods_to_diff(line_content["mods"])]),
                        diff=mods_to_diff(line_content["mods"]),
                        message=line_content["message"],
                        idx=i,
                    )
