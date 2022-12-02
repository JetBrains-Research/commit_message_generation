from typing import Iterator

import jsonlines
from torch.utils.data import IterableDataset, get_worker_info

from src.utils import CommitTextExample, mods_to_diff


class RetrievalDataset(IterableDataset):
    """Defines a dataset for retrieval approach to commit message generation task as IterableDataset"""

    def __init__(self, data_filename: str):
        self._data_filename = data_filename

    def __iter__(self) -> Iterator[CommitTextExample]:
        with jsonlines.open(self._data_filename) as reader:

            for i, line_content in enumerate(reader):
                info = get_worker_info()
                if i % info.num_workers == info.id:  # type: ignore[union-attr]
                    if "diff" in line_content:
                        diff = line_content["diff"]
                    else:
                        diff = mods_to_diff(line_content["mods"])

                    yield CommitTextExample(diff=diff, message=line_content["message"], idx=i)
