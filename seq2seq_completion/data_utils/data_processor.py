import re
import itertools
import torch
from typing import List, Union, Optional, Dict
from string import punctuation
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class DataProcessor:
    """
    This class prepares input for generation pipeline.
    """

    def __init__(
        self,
        prompt_max_len: int,
        diff_tokenizer_name_or_path: str,
        msg_tokenizer_name_or_path: str,
        preprocessing: bool = False,
        nl_token: str = "\n",
    ):
        self._prompt_max_len = prompt_max_len
        self._diff_tokenizer = AutoTokenizer.from_pretrained(diff_tokenizer_name_or_path, use_fast=True)
        self._msg_tokenizer = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path, use_fast=True)
        self._nl_token = nl_token  # this might not be needed? (newline char in input is most likely \n by default)
        self._preprocessing = preprocessing

    def __call__(
        self, decoder_context: str, prefix: Optional[str] = None, diff: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        processed_input = {
            "decoder_input_ids": self.prepare_decoder_input(decoder_context, prefix),
            "encoder_input_ids": torch.empty(0),
        }
        if diff:
            processed_input["encoder_input_ids"] = self.prepare_encoder_input(diff)
        return processed_input

    def prepare_encoder_input(self, diff: str) -> torch.Tensor:
        """
        This method prepares encoder input (diff)r:
        1) (optional) Preprocesses diff
        2) Tokenizes diff
        """
        if self._preprocessing:
            diff = self.preprocess_diff(diff)
        tokenized_diff = self.tokenize(diff, self._diff_tokenizer, truncation=True, max_length=500)
        return tokenized_diff

    def prepare_decoder_input(self, decoder_context: str, prefix: Optional[str]) -> torch.Tensor:
        """
        This method prepares decoder input (consisting of history & message prefix):
        1) (optional) Preprocesses input
        2) Removes last occurence of prefix
        3) Tokenizes input
        4) Adds <bos> token at the beginning
        """
        if self._preprocessing:
            decoder_context = self.preprocess_msg(decoder_context)

        if prefix:
            if " " + decoder_context.split()[-1] != prefix:
                raise ValueError(
                    "if `prefix` is defined, it is expected to be the last word in `decoder_context` "
                    "(with leading space)"
                )
            decoder_context = decoder_context[: decoder_context.rfind(prefix)]

        tokenized_decoder_context = self.tokenize(decoder_context, self._msg_tokenizer)[:, -self._prompt_max_len :]
        tokenized_decoder_context = torch.cat(
            (
                torch.ones(1, 1) * self._msg_tokenizer.bos_token_id,
                tokenized_decoder_context,
            ),
            dim=1,
        )
        return tokenized_decoder_context.long()

    def preprocess_diff(self, diff: str) -> str:
        """
        This method preprocessed single diff string.
        Currently _preprocessing for diffs includes the following:
            - padding punctuation with spaces
            - removing some unnecessary special info
            - removing non-changed lines
        """
        diff = re.sub("([" + punctuation + "\n\t\r])", r" \1 ", diff)
        diff = re.sub("< FILE >", "<FILE>", diff)
        diff = re.sub("< nl >", "<nl>", diff)
        diff_lines = [line.split() for line in diff.split(self._nl_token)]
        processed_lines = []

        for line in diff_lines:
            if len(line) == 0:
                # remove empty lines
                continue

            elif line[0] == "<FILE>":
                # name of changed file
                # example: <FILE> telecomm / java / android / telecomm / Connection . java
                processed_lines.append(line[1:])

            elif line[:2] == ["new", "file"]:
                # line in git diff when new file is created
                # example: new file
                processed_lines.append(["new", "file"])

            elif line[:2] == ["deleted", "file"]:
                # line in git diff when file is deleted
                # example: deleted file
                processed_lines.append(["deleted", "file"])

            elif line[:2] == ["rename", "from"]:
                # line in git diff when file was renamed (old name)
                # example: rename from src / forge / resources / worldedit . properties
                processed_lines.append(line)

            elif line[:2] == ["rename", "to"]:
                # line in git diff when file was renamed (new name)
                # example: rename to src / forge / resources / defaults / worldedit . properties
                processed_lines.append(line)

            elif line[0] == "-":
                # lines that were removed
                # example: - version = ' 2 . 0 . 2 '
                processed_lines.append(line)

            elif line[0] == "+":
                # lines that were added
                # example: + version = ' 2 . 0 . 3 '
                processed_lines.append(line)

            elif (
                line[0] == "index"
                or line[:2] == ["similarity", "index"]
                or (line[:2] == ["@", "@"] and line[-2:] == ["@", "@"])
            ):
                # some special info that we are not interested in
                # example 1: index 0000000 . . 3f26e45
                # example 2: similarity index 100 %
                # example 3: @ @ - 0 , 0 + 1 , 192 @ @
                continue

            else:
                # all other cases
                # case 1: line that was not changed (drop them)
                # case 2: Binary files a / dependencies / windows / sumatra / SumatraPDF . exe and / dev / null differ
                if line[:2] == ["Binary", "files"]:
                    processed_lines.append(line)

        processed_diff = " ".join(itertools.chain(*[line + ["\n"] for line in processed_lines]))
        return processed_diff

    def preprocess_msg(self, msg: str) -> str:
        """
        This method preprocessed single message string.
        Currently _preprocessing for messages includes the following:
            - padding punctuation with spaces
            - making sure that newline character is \n

        (not very useful but we might want to add more sophisticated _preprocessing later?)
        """
        msg = re.sub("([" + punctuation + "\n\t\r])", r" \1 ", msg)
        msg = re.sub("< nl >", "<nl>", msg)
        msg = re.sub(r" +", " ", msg)
        msg = msg.strip(" ")
        return msg.replace(self._nl_token, "\n")

    def tokenize(
        self, inputs: Union[str, List[str]], tokenizer: PreTrainedTokenizerBase, **tokenizer_kwargs
    ) -> torch.Tensor:
        """
        This method tokenizes input string(s) via tokenizer from Transformers.
        Currently tokenization is the same for diffs and messages (except for tokenizers),
        so this method can be used for either of them.
        """
        # in case with empty list just return empty tensor
        if not inputs:
            return torch.empty(0)
        tokenized_inputs = tokenizer(
            inputs, padding=False, return_tensors="pt", return_attention_mask=False, **tokenizer_kwargs
        )
        return tokenized_inputs.input_ids
