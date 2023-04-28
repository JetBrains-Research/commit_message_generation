import logging
import os
from typing import Any, Callable, Dict, List, Optional

import jsonlines
from tiktoken import Encoding
from tqdm import tqdm

from .cmg_prompts import CMGChatPrompts, CMGPrompts


class DataPreprocessor:
    prompt_constructors: Dict[str, Callable] = {
        "simple": CMGPrompts.zero_shot_simple,
        "history": CMGPrompts.zero_shot_history,
    }
    prompt_constructors_chat: Dict[str, Callable] = {
        "simple": CMGChatPrompts.zero_shot_simple,
        "history": CMGChatPrompts.zero_shot_history,
    }

    def __init__(self, tokenizer: Encoding, max_number_of_tokens: int, prompt_configuration: str, use_chat: bool):
        self._tokenizer = tokenizer
        self._max_number_of_tokens = max_number_of_tokens
        self._prompt_configuration = prompt_configuration
        self._use_chat = use_chat

    @staticmethod
    def _process_mods(mods: List[Dict[str, str]]) -> str:
        """
        Transforms a list of all files modifications made in a commit into a single string representation.

        Specifically, adds a header to each file diff (https://git-scm.com/docs/git-diff#_generating_patch_text_with_p)
        and concatenates the results.

        Args:
            mods: A list of file modifications made in a commit.

        Returns:
            A single string representation of all file modifications made in a commit.
        """
        diff = ""

        for mod in mods:
            if mod["change_type"] == "UNKNOWN":
                continue
            elif mod["change_type"] == "ADD":
                file_diff = f"new file {mod['new_path']}"
            elif mod["change_type"] == "DELETE":
                file_diff = f"deleted file {mod['old_path']}"
            elif mod["change_type"] == "RENAME":
                file_diff = f"rename from {mod['old_path']}\nrename to {mod['new_path']}"
            elif mod["change_type"] == "COPY":
                file_diff = f"copy from {mod['old_path']}\ncopy to {mod['new_path']}"
            else:
                file_diff = f"{mod['new_path']}"

            diff += file_diff + "\n" + mod["diff"] + "\n"

        return diff

    def _truncate_diff(self, diff: str) -> str:
        encoding = self._tokenizer.encode(diff)
        if len(encoding) > self._max_number_of_tokens:
            diff = self._tokenizer.decode(encoding[: self._max_number_of_tokens])
        return diff

    def process(
        self,
        mods: List[Dict[str, str]],
        prompt_configuration: str,
        previous_message: Optional[str] = None,
        prefix: str = "",
    ) -> str:
        # get diff from a list of file modifications (combine + add heading in a git fashion)
        diff = DataPreprocessor._process_mods(mods)

        # truncate diff
        diff = self._truncate_diff(diff)

        # construct a prompt for Completion based on the diff (and possibly the previous message)
        prompt = DataPreprocessor.prompt_constructors[prompt_configuration](
            diff=diff, previous_message=previous_message, prefix=prefix
        )
        return prompt

    def process_chat(
        self,
        mods: List[Dict[str, str]],
        prompt_configuration: str,
        previous_message: Optional[str] = None,
        prefix: str = "",
    ) -> List[Dict[str, str]]:
        # get diff from a list of file modifications (combine + add heading in a git fashion)
        diff = DataPreprocessor._process_mods(mods)

        # truncate diff
        diff = self._truncate_diff(diff)

        # construct an input for ChatCompletion based on the diff (and possibly the previous message)
        chat_messages = DataPreprocessor.prompt_constructors_chat[prompt_configuration](
            diff=diff, previous_message=previous_message, prefix=prefix
        )
        return chat_messages

    def process_file(
        self, input_path: str, output_path: str, chunksize: int, use_cache: bool, context_ratio: float
    ) -> None:
        if use_cache and os.path.exists(output_path):
            logging.info("Found preprocessed prompts!")
        else:
            logging.info("Start processing prompts!")
            open(output_path, "w").close()

            chunk: List[Dict[str, Any]] = []
            with jsonlines.open(input_path, "r") as reader:
                for line in tqdm(reader, "Processing prompts from input file"):
                    if len(chunk) > chunksize:
                        with jsonlines.open(output_path, "a") as writer:
                            writer.write_all(chunk)
                        chunk = []

                    if "previous_message" in line:
                        previous_message = line["previous_message"]
                    else:
                        previous_message = None

                    if context_ratio == 0.0:
                        prefix, target = "", line["message"]
                    else:
                        context_len = int(context_ratio * len(line["message"]))
                        prefix, target = line["message"][:context_len], line["message"][context_len:]
                    if self._use_chat:
                        prompt = {
                            "prompt": None,
                            "messages": self.process_chat(
                                mods=line["mods"],
                                previous_message=previous_message,
                                prefix=prefix,
                                prompt_configuration=self._prompt_configuration,
                            ),
                            "target": target,
                        }
                    else:
                        prompt = {
                            "prompt": self.process(
                                mods=line["mods"],
                                previous_message=previous_message,
                                prefix=prefix,
                                prompt_configuration=self._prompt_configuration,
                            ),
                            "messages": None,
                            "target": target,
                        }
                    chunk.append(prompt)

                if len(chunk) > 0:
                    with jsonlines.open(output_path, "a") as writer:
                        writer.write_all(chunk)
            logging.info("Finish processing prompts!")
