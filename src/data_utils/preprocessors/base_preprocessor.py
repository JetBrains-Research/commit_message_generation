import json
import logging
import os
import random
from abc import ABC, abstractmethod
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
        Aggregates commit message history for each author in given file.

        Input file should be in JSONL format and contain keys "author" and "msg_input_ids".
        Also, messages from each author are expected to be in chronological order
        (it won't break anything, but logically result would be incorrect).

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
        history = df[["author", "msg_input_ids"]].groupby("author").agg(list)["msg_input_ids"].to_dict()
        history = {int(key): history[key] for key in history}
        with open(output_path, "w") as f:
            json.dump(history, f)

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
        return chunk

    def process(
        self, input_dir: str, data_dir: str, part: str, message_kwargs: Dict[str, Any], diff_kwargs: Dict[str, Any]
    ) -> None:
        """
        Main processing logic.

        1. Iterate over input file in chunks, process and tokenize messages and diffs, save to separate file.
        2. Aggregate history from processed file, save to separate file.
        3. If processing train, shuffle processed file and save to yet another separate file.

        Args:
            input_dir: Path to directory with input files.
            data_dir: Path to directory with processed files.
            part: Current dataset part.
            message_kwargs: Arbitrary keyword arguments for message processing.
            diff_kwargs: Arbitrary keyword arguments for diffs processing.
        """
        input_path = os.path.join(input_dir, f"{part}.jsonl")
        processed_path = os.path.join(data_dir, f"{part}_processed.jsonl")

        if os.path.exists(processed_path):
            logging.warning(f"Rewriting {processed_path}")
            open(processed_path, "w").close()

        logging.info(f"Processing {input_path} in chunks")
        reader = pd.read_json(input_path, orient="records", lines=True, chunksize=self._chunksize)
        for chunk in tqdm(reader, leave=False):
            processed_chunk = self._process_chunk(chunk, message_kwargs, diff_kwargs)
            with jsonlines.open(processed_path, "a") as writer:
                writer.write_all(
                    processed_chunk[
                        ["author", "pos_in_history", "message", "msg_input_ids", "diff_input_ids", "id"]
                    ].to_dict(orient="records")
                )

        logging.info("Processing history")
        self._process_history(input_path=processed_path, output_path=os.path.join(data_dir, f"{part}_history.json"))

        if part == "train":
            logging.info("Shuffling train")
            with jsonlines.open(processed_path, "r") as reader:
                data = [line for line in reader]
            random.shuffle(data)
            with jsonlines.open(os.path.join(data_dir, f"{part}_shuffled.jsonl"), "w") as writer:
                writer.write_all(data)

    def process_retrieved(self, retrieved_dir: str, data_dir: str, part: str) -> None:
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

        logging.info(f"Processing {retrieved_input_fname}")
        open(retrieved_output_fname, "w").close()
        with jsonlines.open(retrieved_input_fname, "r") as reader:
            for pred in reader:
                retrieved_example = json.loads(getline(input_fname, pred["pos_in_file"] + 1).rstrip("\n"))
                retrieved_example["distance"] = pred["distance"]
                with jsonlines.open(retrieved_output_fname, "a") as writer:
                    writer.write(retrieved_example)

        if part == "train":
            logging.info("Shuffling train")
            with jsonlines.open(retrieved_output_fname, "r") as reader:
                data = [line for line in reader]
            random.shuffle(data)
            with jsonlines.open(os.path.join(data_dir, f"retrieved_{part}_shuffled.jsonl"), "w") as writer:
                writer.write_all(data)
