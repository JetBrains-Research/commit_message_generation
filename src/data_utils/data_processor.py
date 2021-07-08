import re
import itertools
import torch
from typing import List, Union, Optional, Dict
from string import punctuation
from transformers import AutoTokenizer, PreTrainedTokenizerBase  # type: ignore


class DataProcessor:
    """
    This class prepares input for generation pipeline.
    """

    def __init__(
        self,
        prompt_max_len: int,
        diff_tokenizer_name_or_path: str,
        msg_tokenizer_name_or_path: str,
        nl_token: str = "\n",
    ):
        self.prompt_max_len = prompt_max_len
        self.diff_tokenizer = AutoTokenizer.from_pretrained(diff_tokenizer_name_or_path, use_fast=True)
        self.msg_tokenizer = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path, use_fast=True)
        self.nl_token = nl_token  # this might not be needed? (newline char in input is most likely \n by default)

    def __call__(self, msg: str, history: List[str], diff: Optional[str] = None) -> Dict[str, torch.Tensor]:
        processed_input = {
            "decoder_input_ids": self.prepare_decoder_input(msg, history),
            "encoder_input_ids": torch.empty(0),
        }
        if diff:
            processed_input.update({"encoder_input_ids": self.prepare_encoder_input(diff)})
        return processed_input

    def prepare_encoder_input(self, diff: str) -> torch.Tensor:
        """
        This method prepares input (diff) for encoder:
        1) Preprocesses diff
        2) Tokenizes diff
        """
        preprocessed_diff = self.preprocess_diff(diff)
        tokenized_diff = self.tokenize(preprocessed_diff, self.diff_tokenizer)
        return tokenized_diff

    def prepare_decoder_input(self, msg: str, history: List[str]) -> torch.Tensor:
        """
        This method prepares input (message & history) for decoder:
        1) Preprocesses message and history
        2) Tokenizes message and history
        3) Concatenates history with message
        """
        preprocessed_msg = self.preprocess_msg(msg)
        preprocessed_history = [self.preprocess_msg(old_msg) for old_msg in history]

        tokenized_msg = self.tokenize(preprocessed_msg, self.msg_tokenizer)
        tokenized_history = self.tokenize(preprocessed_history, self.msg_tokenizer)
        return self.concat_history_and_msg(tokenized_msg, tokenized_history)

    def preprocess_diff(self, diff: str) -> str:
        """
        This method preprocessed single diff string.
        Currently preprocessing for diffs includes the following:
            - padding punctuation with spaces
            - removing some unnecessary special info
            - removing non-changed lines
        """
        diff = re.sub("([" + punctuation + "\n\t\r])", r" \1 ", diff)
        diff = re.sub("< FILE >", "<FILE>", diff)
        diff_lines = [line.split() for line in diff.split(self.nl_token)]
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
        Currently preprocessing for messages includes the following:
            - padding punctuation with spaces
            - making sure that newline character is \n

        (not very useful but we might want to add more sophisticated preprocessing later?)
        """
        msg = re.sub("([" + punctuation + "\n\t\r])", r" \1 ", msg)
        msg = re.sub(r" +", " ", msg)
        msg = msg.strip(" ")
        return msg.replace(self.nl_token, "\n")

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
            inputs, padding=False, truncation=True, return_tensors="pt", return_attention_mask=False, **tokenizer_kwargs
        )
        return tokenized_inputs.input_ids

    def concat_history_and_msg(self, msg: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """
        This method concatenates history with current message.
        Resulting generation prompt looks the following way:
        - <bos> history_1 \n ... history_k \n msg <eos>
        """
        msg = msg[:, : self.prompt_max_len - 2]  # truncate message if necessary
        cur_len = msg.shape[1]
        # insert previous messages from history until we reach max_len
        for old_msg in torch.flip(history, dims=[0]):
            if cur_len + old_msg.shape[0] + len(self.msg_tokenizer(" \n ").input_ids) > self.prompt_max_len - 2:
                break
            msg = torch.cat((old_msg.unsqueeze(0), self.tokenize(" \n ", self.msg_tokenizer), msg), dim=1)
            cur_len = msg.shape[1]

        # add <bos> and <eos> tokens
        msg = torch.cat(
            (
                torch.ones(1, 1) * self.msg_tokenizer.bos_token_id,
                msg,
                torch.ones(1, 1) * self.msg_tokenizer.eos_token_id,
            ),
            dim=1,
        )
        return msg.long()
