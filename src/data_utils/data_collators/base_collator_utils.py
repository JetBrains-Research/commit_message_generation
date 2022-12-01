from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizerFast

from src.utils import SingleExample


@dataclass
class BaseCollatorUtils:
    """Base class for utilities both for training and evaluations collators (e.g. processing encoder input).

    Args:
        diff_tokenizer: Tokenizer used to tokenize diff.
        msg_tokenizer: Tokenizer used to tokenize messages.
        encoder_context_max_len: Maximum allowed number of tokens in encoder context.
        decoder_context_max_len: Maximum allowed number of tokens in decoder context.
        with_history: True to add history to decoder input, False otherwise.
        encoder_input_type: Should be one of `history`, `diff`, corresponding data will be used
          to construct encoder input.
        testing: True to generate tensors of maximum possible shape with random numbers instead of actually processing
         input data  (used to quickly test whether current batch size fits in GPU memory).
    """

    diff_tokenizer: PreTrainedTokenizerFast
    msg_tokenizer: PreTrainedTokenizerFast
    encoder_context_max_len: int
    decoder_context_max_len: int
    with_history: bool
    encoder_input_type: str
    testing: bool

    def _pad_tensor(self, input_tensor: torch.Tensor, pad_len: int, value: int, left: bool) -> torch.Tensor:
        return torch.nn.functional.pad(
            input_tensor, pad=[pad_len, 0] if left else [0, pad_len], mode="constant", value=value
        )

    def _get_history(self, cur_len: int, history_ids: List[List[int]]) -> List[List[int]]:
        """
        A helper method to use history in decoder's context.

        It iterates over history starting from the most recent message and adds messages until total length exceeds
        decoder context length.

        Args:
            cur_len: Length of corresponding message, because history + message should fit to decoder context.
            history_ids: All messages from history, tokenized.

        Returns:
            A subset of history_ids that in total with cur_len
            won't exceed maximum allowed decoder context len.
        """
        cur_history_ids = []
        for history_input_ids in history_ids[::-1]:
            if cur_len + len(history_input_ids) + 1 > self.decoder_context_max_len:
                break

            cur_len += len(history_input_ids) + 1
            cur_history_ids.append(history_input_ids + [self.msg_tokenizer.sep_token_id])  # type: ignore[attr-defined]
        return cur_history_ids[::-1]

    def _process_history(self, history_inputs: List[List[List[int]]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This helper method processes history as encoder input.

        It iterates over history starting from the most recent message and adds messages until total length exceeds
        encoder context length.

        It also adds all required special tokens: format is [BOS] history_1 [SEP] ... [SEP] history_k [EOS]

        Finally, it is responsible for padding to maximum length in batch and conversion to torch.Tensor.

        Args:
            history_inputs: A list of histories for current batch.

        Returns:
            input_ids for encoder, attention_mask for encoder
        """
        all_history_ids: List[torch.Tensor] = []
        all_history_masks: List[torch.Tensor] = []

        for cur_example_history in history_inputs:
            cur_history_ids = []
            cur_len = 2
            for history_ids in cur_example_history[::-1]:
                if cur_len + len(history_ids) + 1 > self.encoder_context_max_len:
                    break

                cur_len += len(history_ids) + 1
                cur_history_ids.append(history_ids + [self.msg_tokenizer.sep_token_id])  # type: ignore[attr-defined]

            cur_history_ids = (
                [[self.msg_tokenizer.bos_token_id]] + cur_history_ids[::-1] + [[self.msg_tokenizer.eos_token_id]]  # type: ignore[attr-defined]
            )
            # drop last [SEP] token
            cur_history_ids[-2] = cur_history_ids[-2][:-1]

            cur_history_ids_tensor = torch.tensor(
                [ex for sublist in cur_history_ids for ex in sublist], dtype=torch.int64
            )
            cur_history_mask_tensor = torch.ones_like(cur_history_ids_tensor)

            all_history_ids.append(cur_history_ids_tensor)
            all_history_masks.append(cur_history_mask_tensor)

        history_max_len = max(len(tensor) for tensor in all_history_ids)

        # pad tensors to max length in batch
        all_history_ids = [
            self._pad_tensor(
                tensor, pad_len=history_max_len - tensor.numel(), value=self.msg_tokenizer.pad_token_id, left=False  # type: ignore[attr-defined]
            )
            for tensor in all_history_ids
        ]
        all_history_masks = [
            self._pad_tensor(tensor, pad_len=history_max_len - tensor.numel(), value=0, left=False)
            for tensor in all_history_masks
        ]
        return torch.stack(all_history_ids), torch.stack(all_history_masks)

    def _process_diff(self, diff_inputs: List[List[int]]):
        """
        This helper method processes history as encoder input.

        It truncates the diffs to maximum allowed length.

        It also adds all required special tokens: format is [BOS] diff [EOS].

        Finally, it is responsible for padding to maximum length in batch and conversion to torch.Tensor.

        Args:
            diff_inputs: A list of diffs for current batch.

        Returns:
            input_ids for encoder, attention_mask for encoder
        """
        all_diff_ids = [
            [self.diff_tokenizer.bos_token_id]  # type: ignore[attr-defined]
            + diff[: self.encoder_context_max_len - 2]
            + [self.diff_tokenizer.eos_token_id]  # type: ignore[attr-defined]
            for diff in diff_inputs
        ]
        all_diff_ids_tensors = [torch.tensor(ids, dtype=torch.int64) for ids in all_diff_ids]
        all_diff_masks_tensors = [torch.ones_like(ids) for ids in all_diff_ids_tensors]

        # pad tensors to max length in batch
        diff_max_len = max(len(tensor) for tensor in all_diff_ids)
        all_diff_ids_tensors = [
            self._pad_tensor(
                tensor,
                pad_len=diff_max_len - tensor.numel(),
                value=self.diff_tokenizer.pad_token_id,  # type: ignore[attr-defined]
                left=False,
            )
            for tensor in all_diff_ids_tensors
        ]
        all_diff_masks_tensors = [
            self._pad_tensor(
                tensor,
                pad_len=diff_max_len - tensor.numel(),
                value=0,
                left=False,
            )
            for tensor in all_diff_masks_tensors
        ]
        return torch.stack(all_diff_ids_tensors), torch.stack(all_diff_masks_tensors)

    def _process_encoder_input(self, examples: List[SingleExample]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        A helper method to process encoder input.

        Either diff or history can be passed to encoder.

        Args:
            examples: A batch of examples from dataset.

        Returns:
            input_ids for encoder, attention_mask for encoder
        """
        if self.encoder_input_type == "diff":
            diff_inputs: List[List[int]] = [example.diff_input_ids for example in examples]
            return self._process_diff(diff_inputs)
        elif self.encoder_input_type == "history":
            history_inputs: List[List[List[int]]] = [example.history_input_ids for example in examples]
            return self._process_history(history_inputs)
        else:
            raise ValueError("Unknown encoder input type. Currently supported are `diff` and `history`.")
