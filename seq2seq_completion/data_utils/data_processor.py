import re
import torch
from typing import List, Union, Optional, Dict
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
    ):
        self._prompt_max_len = prompt_max_len
        self._diff_tokenizer = AutoTokenizer.from_pretrained(diff_tokenizer_name_or_path, use_fast=True)
        self._msg_tokenizer = AutoTokenizer.from_pretrained(msg_tokenizer_name_or_path, use_fast=True)
        self._preprocessing = preprocessing

    def __call__(self, decoder_context: str, diff: Optional[str] = None) -> Dict[str, torch.Tensor]:
        processed_input = {
            "decoder_input_ids": self.prepare_decoder_input(decoder_context),
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

    def prepare_decoder_input(self, decoder_context: str) -> torch.Tensor:
        """
        This method prepares decoder input (consisting of history & message prefix):
        1) Tokenizes input
        2) Adds <bos> token at the beginning
        """
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
            - removing some unnecessary special info
            - removing non-changed lines
        """
        diff_lines = diff.split("\n")
        processed_lines = []

        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]

            if line.startswith("diff --git"):
                # header line in git diff
                # example: diff --git a/.gitignore b/.gitignore
                if not (
                    diff_lines[i + 1].startswith("new file")
                    or diff_lines[i + 1].startswith("deleted file")
                    or diff_lines[i + 2].startswith("rename")
                ):
                    i += 2
                    processed_lines.append(line.split()[2][2:])

            elif line.startswith("new file"):
                # line in git diff when file was created
                # example: new file mode <mode>
                # --- /dev/null (date <smth>)
                # +++ b/<filename> (date <smth>)
                filename = diff_lines[i + 2].split()[1][2:]
                i += 2
                processed_lines.append(f"new file {filename}")

            elif line.startswith("deleted file"):
                # line in git diff when file was created or deleted
                # example: deleted file mode <mode>
                # --- a/<filename> (date <smth>)
                # +++ /dev/null (date <smth>)
                filename = diff_lines[i + 1].split()[1][2:]
                i += 2
                processed_lines.append(f"deleted file {filename}")

            elif line.startswith("rename"):
                # lines in git diff when file was renamed
                # example: rename from <old_filename>, rename to <new_filename>
                processed_lines.append(line)

            elif line.startswith("-") and len(line.strip()) > 1:
                # lines that were removed
                # example: - version='2.0.2', -version='2.0.2'
                processed_lines.append(line)

            elif line.startswith("+") and len(line.strip()) > 1:
                # lines that were added
                # example: + version='2.0.2', +version='2.0.2
                processed_lines.append(line)

            elif line.startswith("Binary files") and line.endswith("differ"):
                # example: Binary files <file1> and <file2> differ
                processed_lines.append(line)

            i += 1

        processed_diff = "\n".join(processed_lines)
        processed_diff = re.sub(r" +", " ", processed_diff)
        return processed_diff

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
