import json
import linecache
import logging
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from linecache import getline
from typing import Any, Dict, List, Optional

import jsonlines
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


class BasePreprocessor(ABC):
    """Base class for data preprocessing.

    Implements common logic such as reading/writing files and tokenizing messages.
    Subclasses should implement logic for processing individual messages and diffs (modifications).
    """

    def __init__(
        self,
        diff_tokenizer: PreTrainedTokenizerFast,
        msg_tokenizer: PreTrainedTokenizerFast,
        chunksize: Optional[int] = None,
    ):
        super().__init__()
        self._diff_tokenizer = diff_tokenizer
        self._msg_tokenizer = msg_tokenizer
        self._chunksize = chunksize if chunksize else 4096
        self._num_commits: Dict[int, int] = defaultdict(int)

    @abstractmethod
    def _preprocess_mods(self, mods: List[Dict[str, str]], **kwargs) -> str:
        pass

    @abstractmethod
    def _preprocess_message(self, message: str, **kwargs) -> str:
        pass

    def _tokenize_diffs(self, diffs: List[str], max_len: int = 512) -> List[List[int]]:
        """Tokenizes diffs via transformers' tokenizer.

        Diffs are truncated to save memory. Special tokens are added later, during batch construction, so 2 extra tokens
        from max_length are reserved for BOS and EOS.
        """
        return self._diff_tokenizer(
            diffs, truncation=True, max_length=max_len - 2, padding=False, add_special_tokens=False
        ).input_ids  # type: ignore[operator]

    def _tokenize_messages(self, messages: List[str]) -> List[List[int]]:
        return self._msg_tokenizer(
            messages, truncation=False, padding=False, add_special_tokens=False  # type: ignore[operator]
        ).input_ids

    def _process_history(self, input_path: str, output_path: str) -> None:
        """
        Aggregates commit message history for each author in a given file.

        Input file should be in JSONL format and contain keys "author" and "msg_input_ids".
        Also, messages from each author are expected to be in chronological order
        (it won't break anything, but logically the result would be incorrect).

        Output is a JSON file with authors ids as keys and lists of their messages as values.

        Args:
            input_path: Path to file to read data from.
            output_path: Path to file to save history to.
        """
        data = []
        with jsonlines.open(input_path, "r") as reader:
            for row in reader:
                assert "author" in row
                assert "msg_input_ids" in row
                data.append({"author": int(row["author"]), "msg_input_ids": row["msg_input_ids"]})
        df = pd.DataFrame(data)
        history_records = (
            df.groupby("author").agg(msg_input_ids=("msg_input_ids", list)).reset_index().to_dict(orient="records")
        )
        history = {row["author"]: row["msg_input_ids"] for row in history_records}
        with open(output_path, "w") as f:
            json.dump(history, f)

    def _add_history_to_inputs(
        self, input_path: str, history_path: str, output_path: str, decoder_context_max_length: int, part: str
    ) -> None:
        """Adds commit message history to each example in the input file and saves the results to the output file.

        This approach uses more disk space but enables working with the dataset in a fully iterable fashion
        without loading the history into RAM. To prevent excessive disk usage, the messages from history are added only
        until the maximum decoder context length is achieved.

        Args:
            input_path: Path to file to read data from.
            history_path: Path to file to read history from.
            output_path: Path to file to save data with history inputs to.
            decoder_context_max_length: Maximum allowed number of tokens in decoder context.
            part: Current dataset part.
        """
        with open(history_path, "r") as f:
            history = json.load(f)

        open(output_path, "w").close()
        with jsonlines.open(input_path, "r") as reader:
            for line in tqdm(reader, desc=f"Aggregating history inputs for {part}"):
                all_author_history: List[List[int]] = history[str(line["author"])][: line["pos_in_history"]]
                relevant_author_history: List[List[int]] = []
                cur_len = len(line["msg_input_ids"]) + 2
                for history_msg in all_author_history[::-1]:
                    if cur_len + len(history_msg) + 1 > decoder_context_max_length:
                        break
                    relevant_author_history.append(history_msg)
                    cur_len += len(history_msg) + 1
                line["history_input_ids"] = relevant_author_history[::-1]

                with jsonlines.open(output_path, "a") as writer:
                    writer.write(line)

    def _get_pos_in_history(self, authors: List[int]) -> List[int]:
        """Builds correct position in history for each commit when iterating over input data
        in chunks.

        Args:
            authors: A list of authors for commits from the current chunk.

        Returns:
            A list of positions in the corresponding author's history for each commit from the chunk.
        """
        positions_in_history = []
        for author in authors:
            self._num_commits[author] += 1
            positions_in_history.append(self._num_commits[author] - 1)
        return positions_in_history

    def _process_chunk(
        self, chunk: pd.DataFrame, message_kwargs: Dict[str, Any], diff_kwargs: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Processes a single chunk, which includes:

        * processing messages
        * processing diffs (modifications)
        * tokenizes messages
        * tokenizes diffs

        Args:
            chunk: Data to process.
            message_kwargs: Arbitrary keyword arguments for message processing.
            diff_kwargs: Arbitrary keyword arguments for diffs processing.
              Should also include "max_length" for tokenization.
        """
        assert "max_len" in diff_kwargs and isinstance(diff_kwargs["max_len"], int)
        chunk["message"] = [
            self._preprocess_message(example, **message_kwargs) for _, example in chunk["message"].items()
        ]
        chunk["mods"] = [self._preprocess_mods(example, **diff_kwargs) for _, example in chunk["mods"].items()]
        chunk["msg_input_ids"] = self._tokenize_messages(chunk["message"].tolist())
        chunk["diff_input_ids"] = self._tokenize_diffs(chunk["mods"].tolist(), max_len=diff_kwargs["max_len"])
        chunk["pos_in_history"] = self._get_pos_in_history(chunk["author"].tolist())
        return chunk

    def _shuffle(self, input_path: str, output_path: str) -> None:
        """Shuffles a file.

        To support moderately large files, it works by shuffling a list of line idxs
        and then utilizing `linecache` to write specific lines in a new order.

        Args:
            input_path: Path to input file.
            output_path: Path to output file.
        """
        random.seed(42)
        logging.info("Calculating number of lines")
        with open(input_path) as f:
            num_lines = sum(1 for _ in f)
        logging.info("Shuffling line idxs")
        idxs = [i + 1 for i in range(num_lines)]  # start rows idxs with 1, since linecache starts with 1
        random.shuffle(idxs)
        with open(output_path, "w") as f:
            for i in tqdm(idxs, f"Writing shuffled lines for {input_path}..."):
                f.write(linecache.getline(input_path, i))

    def process(
        self,
        input_dir: str,
        data_dir: str,
        part: str,
        message_kwargs: Dict[str, Any],
        diff_kwargs: Dict[str, Any],
        use_cache: bool,
        add_history_to_inputs: bool,
        decoder_context_max_length: Optional[int] = None,
    ) -> None:
        """
        Main processing logic.

        1. Iterate over input file in chunks, process and tokenize messages and diffs, save to separate file.
        2. Aggregate history from processed file, save to separate file.
        3. If add_history_to_inputs is True, iterate through input file again and aggregate history inputs for each example.
        4. If processing train, shuffle processed file and save to yet another separate file.

        Args:
            input_dir: Path to directory with input files.
            data_dir: Path to directory with processed files.
            part: Current dataset part.
            message_kwargs: Arbitrary keyword arguments for message processing.
            diff_kwargs: Arbitrary keyword arguments for diffs processing.
            use_cache: True to use already processed files when possible, False otherwise.
            add_history_to_inputs: True to add history inputs to each example in a processed file.
            decoder_context_max_length: Should be provided when add_history_to_inputs is True.
        """
        input_path = os.path.join(input_dir, f"{part}.jsonl")
        processed_path = os.path.join(data_dir, f"{part}_processed.jsonl")

        if use_cache and os.path.exists(processed_path):
            logging.info(f"{part}_processed.jsonl found, won't rewrite")
        else:
            open(processed_path, "w").close()
            logging.info(f"Processing {input_path} in chunks")
            reader = pd.read_json(input_path, orient="records", lines=True, chunksize=self._chunksize)
            for chunk in tqdm(reader, leave=False):
                processed_chunk = self._process_chunk(chunk, message_kwargs, diff_kwargs)
                with jsonlines.open(processed_path, "a") as writer:
                    writer.write_all(
                        processed_chunk[
                            [
                                "author",
                                "message",
                                "msg_input_ids",
                                "diff_input_ids",
                                "hash",
                                "repo",
                                "language",
                                "pos_in_history",
                            ]
                        ].to_dict(orient="records")
                    )
        if use_cache and os.path.exists(os.path.join(data_dir, f"{part}_history.json")):
            logging.info(f"{part}_history found, won't rewrite")
        else:
            logging.info("Processing history")
            self._process_history(input_path=processed_path, output_path=os.path.join(data_dir, f"{part}_history.json"))
        if add_history_to_inputs:
            if use_cache and os.path.exists(os.path.join(data_dir, f"{part}_processed_history.json")):
                logging.info(f"{part}_processed_history found, won't rewrite")
            else:
                assert (
                    decoder_context_max_length is not None
                ), "You have to define max context length to aggregate history in inputs."
                self._add_history_to_inputs(
                    input_path=processed_path,
                    history_path=os.path.join(data_dir, f"{part}_history.json"),
                    output_path=os.path.join(data_dir, f"{part}_processed_history.jsonl"),
                    part=part,
                    decoder_context_max_length=decoder_context_max_length,
                )
            processed_path = os.path.join(data_dir, f"{part}_processed_history.jsonl")

        if part == "train":
            if use_cache and os.path.exists(os.path.join(data_dir, f"{part}_shuffled.jsonl")):
                logging.info(f"{part}_shuffled found, won't rewrite")
            else:
                logging.info("Shuffling train")
                self._shuffle(input_path=processed_path, output_path=os.path.join(data_dir, f"{part}_shuffled.jsonl"))

    def process_retrieved(self, retrieved_dir: str, data_dir: str, part: str, use_cache: bool) -> None:
        """
        Retrieval processing logic. Should be called after `process`, as it relies on processed files.

        1. Iterate over processed train file, obtain examples in an order specified in retrieved file.
        2. If processing train, shuffle processed file and save to separate file.

        Shuffling should be done with the same random seed as `process`!! `seed_everything` from Lightning should take
        care of this.

        Args:
            retrieved_dir: Path to directory with retrieved files.
            data_dir: Path to directory with processed files.
            part: Current dataset part.

        """
        input_fname = os.path.join(data_dir, "train_processed.jsonl")
        retrieved_input_fname = os.path.join(retrieved_dir, f"{part}_predictions.jsonl")
        retrieved_output_fname = os.path.join(data_dir, f"retrieved_{part}_processed.jsonl")

        if use_cache and os.path.exists(retrieved_output_fname):
            logging.info(f"retrieved_{part}_processed.jsonl found, won't rewrite")
        else:
            logging.info(f"Processing {retrieved_input_fname}")
            open(retrieved_output_fname, "w").close()
            with jsonlines.open(retrieved_input_fname, "r") as reader:
                for pred in reader:
                    retrieved_example = json.loads(getline(input_fname, pred["pos_in_file"] + 1).rstrip("\n"))
                    retrieved_example["distance"] = pred["distance"]
                    with jsonlines.open(retrieved_output_fname, "a") as writer:
                        writer.write(retrieved_example)

        if part == "train":
            if use_cache and os.path.exists(os.path.join(data_dir, f"retrieved_{part}_shuffled.jsonl")):
                logging.info(f"retrieved_{part}_shuffled found, won't rewrite")
            else:
                logging.info("Shuffling train")
                self._shuffle(
                    input_path=retrieved_output_fname,
                    output_path=os.path.join(data_dir, f"retrieved_{part}_shuffled.jsonl"),
                )
