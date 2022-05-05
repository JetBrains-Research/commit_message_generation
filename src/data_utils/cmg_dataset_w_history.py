import json
from typing import Dict, Generator, Iterator, List

import torch
from torch.utils.data import DataLoader, IterableDataset

from .data_collator import DataCollator


class CMGDatasetWithHistory(IterableDataset):
    def __init__(self, filename: str, history: Dict[str, List[int]], rank: int, world_size: int):
        """
        Defines an iterable-style dataset for commit message generation task.
        This version provides history from the same author for each commit.

        Args:
            filename: File to read diff, author ids and positions in history from.
            history: Dictionary with full message history for each author.
            rank: Rank of the process in DDP (must be 0 if you have single process).
            world_size: AAmount of processes in DDP (must be 1 if you have single process).
        """

        self.filename = filename
        self.history = history

        with open(filename, "r") as f:
            self._len = sum(1 for _ in f)

        self._gpu_rank = rank
        self._gpu_world_size = world_size
        self._num_workers = None

        self.world_size = None
        self.process_rank = None

    @staticmethod
    def _init_worker_fn(worker_id: int) -> None:
        """Init each worker for DataLoader in a proper way."""
        worker_info = torch.utils.data.get_worker_info()
        assert worker_id == worker_info.id
        self: CMGDatasetWithHistory = worker_info.dataset
        self.process_rank = self._gpu_rank * self._num_workers + worker_info.id
        self.world_size = self._gpu_world_size * self._num_workers

    def _get_examples_generator(self) -> Generator[Dict[str, List[int]], None, None]:
        """
        For multiprocessing support:

        process_rank = current process id
        world_size = # of processes

        This function yields local_rank'th row from every world_size rows.
        """
        with open(self.filename) as f:
            for i, line in enumerate(f):
                if i % self.world_size == self.process_rank:
                    line = json.loads(line.strip())
                    yield {
                        "diff_input_ids": line["diff_input_ids"],
                        "msg_input_ids": self.history[str(line["author"])][line["pos_in_history"]],
                        "history_input_ids": self.history[str(line["author"])][: line["pos_in_history"]],
                    }

    def __iter__(self) -> Iterator[Dict[str, List[int]]]:
        assert self._num_workers is not None, f"You must access __iter__ through DataLoader"
        return iter(self._get_examples_generator())

    def get_dataloader(self, batch_size: int, num_workers: int, collate_fn: DataCollator) -> DataLoader:
        """Creates DataLoader in a proper way."""
        assert num_workers >= 0, "num_workers must be at least 0"
        if num_workers == 0:
            # We need to initialize at least 1 worker in order to call worker_init_fn
            num_workers = 1
        self._num_workers = num_workers

        return DataLoader(
            dataset=self,
            batch_size=batch_size,  # TODO: https://pytorch.org/docs/stable/data.html#disable-automatic-batching (?)
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=CMGDatasetWithHistory._init_worker_fn,
        )

    @staticmethod
    def load_data(dataset_root: str, rank: int, world_size: int):
        """
        Load dataset from files on disk.

        Args:
            dataset_root: Path to dataset, including part (train/val/test).
            rank: Rank of the process in DDP (must be 0 if you have single process).
            world_size: AAmount of processes in DDP (must be 1 if you have single process).
        """

        with open(dataset_root + "_history.json", "r") as infile:
            history = json.load(infile)

        return CMGDatasetWithHistory(dataset_root + ".json", history, rank, world_size)
