import json
import logging
import os
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import jsonlines
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast


class BasePreprocessor(ABC):
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
        return self._diff_tokenizer(
            diffs, truncation=True, max_length=max_len - 2, padding=False, add_special_tokens=False
        ).input_ids  # type: ignore[operator]

    def _tokenize_messages(self, messages: List[str]) -> List[List[int]]:
        return self._msg_tokenizer(messages, truncation=False, padding=False, add_special_tokens=False).input_ids  # type: ignore[operator]

    def _process_history(self, input_path: str, output_path: str) -> None:
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

    def _process_chunk(self, chunk: pd.DataFrame, message_kwargs, diff_kwargs):
        assert "max_len" in diff_kwargs and isinstance(diff_kwargs["max_len"], int)
        chunk["message"] = [
            self._preprocess_message(example, **message_kwargs) for _, example in chunk["message"].items()
        ]
        chunk["mods"] = [self._preprocess_mods(example, **diff_kwargs) for _, example in chunk["mods"].items()]
        chunk["msg_input_ids"] = self._tokenize_messages(chunk["message"].tolist())
        chunk["diff_input_ids"] = self._tokenize_diffs(chunk["mods"].tolist(), max_len=diff_kwargs["max_len"])
        return chunk

    def process(self, input_dir: str, data_dir: str, part: str, message_kwargs, diff_kwargs):
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
                    processed_chunk[["author", "pos_in_history", "message", "msg_input_ids", "diff_input_ids"]].to_dict(
                        orient="records"
                    )
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
