import json
import logging
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from src.utils import SingleExample


class CMCDatasetWithHistory(IterableDataset):
    def __init__(
        self,
        filename: str,
        history: Dict[str, List[List[int]]],
        rank: int,
        world_size: int,
        retrieval_filename: Optional[str] = None,
    ):
        """
        Defines an iterable-style dataset for commit message completion task.
        This version expects input to be already tokenized and provides history for each commit.

        Args:
            filename: File to read diff, author ids and positions in history from.
            history: Dictionary with full message history for each author.
            rank: Rank of the process in DDP (must be 0 if you have single process).
            world_size: Amount of processes in DDP (must be 1 if you have single process).
            retrieval_filename: File to read retrieved diffs and messages from (optional).
        """

        self._filename = filename
        self._retrieval_filename = retrieval_filename
        self._history = history

        self._len = None

        self._gpu_rank: int = rank
        self._gpu_world_size: int = world_size

        self._num_workers: int
        self._world_size: int
        self._process_rank: int

    def __len__(self):
        if self._len is None:
            logging.info("Calculating length of input file")
            with open(self._filename, "r") as f:
                self._len = sum(1 for _ in f)
        return self._len

    @staticmethod
    def _init_worker_fn(worker_id: int) -> None:
        """Init each worker for DataLoader in a proper way."""
        worker_info = torch.utils.data.get_worker_info()
        assert worker_id == worker_info.id  # type: ignore[union-attr]
        dataset: CMCDatasetWithHistory = worker_info.dataset  # type: ignore[assignment, union-attr]
        dataset._process_rank = dataset._gpu_rank * dataset._num_workers + worker_info.id  # type: ignore[union-attr]
        dataset._world_size = dataset._gpu_world_size * dataset._num_workers

    def _get_examples_generator(self) -> Generator[SingleExample, None, None]:
        """
        For multiprocessing support:

        process_rank = current process id
        world_size = # of processes

        This function yields local_rank'th row from every world_size rows.
        """
        if self._retrieval_filename is None:
            with open(self._filename) as f:
                for i, line in enumerate(f):
                    if i % self._world_size == self._process_rank:
                        example: Dict[str, Any] = json.loads(line.strip())
                        author: str = str(example["author"])
                        diff_input_ids: List[int] = example["diff_input_ids"]
                        pos_in_history: int = example["pos_in_history"]

                        yield SingleExample(
                            diff_input_ids=diff_input_ids,
                            msg_input_ids=self._history[str(author)][pos_in_history],
                            history_input_ids=self._history[str(author)][:pos_in_history],
                        )
        else:
            with open(self._filename) as f:
                with open(self._retrieval_filename) as f_retrieval:
                    for (i, line), (i_retrieval, line_retrieval) in zip(enumerate(f), enumerate(f_retrieval)):

                        assert i == i_retrieval

                        if i % self._world_size == self._process_rank:
                            example: Dict[str, Any] = json.loads(line.strip())  # type: ignore[no-redef]
                            author: str = str(example["author"])  # type: ignore[no-redef]
                            diff_input_ids: List[int] = example["diff_input_ids"]  # type: ignore[no-redef]
                            pos_in_history: int = example["pos_in_history"]  # type: ignore[no-redef]

                            retrieval_example: Dict[str, Any] = json.loads(line_retrieval.strip())
                            retrieved_diff_input_ids: List[int] = retrieval_example["diff_input_ids"]
                            retrieved_msg_input_ids: List[int] = retrieval_example["msg_input_ids"]

                            yield SingleExample(
                                diff_input_ids=diff_input_ids,
                                msg_input_ids=self._history[str(author)][pos_in_history],
                                history_input_ids=self._history[str(author)][:pos_in_history],
                                retrieved_diff_input_ids=retrieved_diff_input_ids,
                                retrieved_msg_input_ids=retrieved_msg_input_ids,
                            )

    def __iter__(self) -> Iterator[SingleExample]:
        assert self._num_workers is not None, f"You must access __iter__ through DataLoader"
        return iter(self._get_examples_generator())

    def get_dataloader(self, batch_size: int, num_workers: int, collate_fn: Callable) -> DataLoader:
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
            worker_init_fn=CMCDatasetWithHistory._init_worker_fn,
        )

    @staticmethod
    def load_data(
        history_path: str,
        data_path: str,
        rank: int,
        world_size: int,
        retrieved_data_path: Optional[str] = None,
    ):
        """
        Load dataset from files on disk.

        Args:
            history_path: Path to history file.
            data_path: Path to data file.
            rank: Rank of the process in DDP (must be 0 if you have single process).
            world_size: Amount of processes in DDP (must be 1 if you have single process).
            retrieved_data_path: Path to retrieved file (optional).
        """

        with open(history_path, "r") as infile:
            history = json.load(infile)

        return CMCDatasetWithHistory(
            filename=data_path,
            retrieval_filename=retrieved_data_path,
            history=history,
            rank=rank,
            world_size=world_size,
        )
