import logging
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch

from src.utils import SingleExample


@dataclass
class BaseCollatorUtils:
    """Base class for utilities both for training and evaluation collators (e.g. processing encoder input).

    Attributes:
        msg_*_token_id: Corresponding special token for message tokenizer.
        diff_*_token_id: Corresponding special token for diff tokenizer.
        encoder_context_max_len: Maximum allowed number of tokens in encoder context.
        decoder_context_max_len: Maximum allowed number of tokens in decoder context.
        with_history: True to add history to decoder input, False otherwise.
        encoder_input_type: Should be one of `history`, `diff`, corresponding data will be used
          to construct encoder input.
        process_retrieved: Whether retrieved examples are expected as input or not.
        testing: True to generate tensors of maximum possible shape with random numbers instead of actually processing
         input data (used to quickly test whether current batch size fits in GPU memory).
    """

    msg_bos_token_id: int
    msg_eos_token_id: int
    msg_pad_token_id: int
    msg_sep_token_id: int
    diff_bos_token_id: int
    diff_eos_token_id: int
    diff_pad_token_id: int
    encoder_context_max_len: int
    decoder_context_max_len: int
    with_history: bool
    encoder_input_type: str
    process_retrieved: bool
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
            cur_history_ids.append(history_input_ids + [self.msg_sep_token_id])
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
                cur_history_ids.append(history_ids + [self.msg_sep_token_id])

            cur_history_ids = [[self.msg_bos_token_id]] + cur_history_ids[::-1] + [[self.msg_eos_token_id]]
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
            self._pad_tensor(tensor, pad_len=history_max_len - tensor.numel(), value=self.msg_pad_token_id, left=False)
            for tensor in all_history_ids
        ]
        all_history_masks = [
            self._pad_tensor(tensor, pad_len=history_max_len - tensor.numel(), value=0, left=False)
            for tensor in all_history_masks
        ]
        return torch.stack(all_history_ids), torch.stack(all_history_masks)

    def _process_inputs(self, inputs: List[List[int]], are_messages: bool = False):
        """
        This helper method processes either diffs or messsages as encoder input.

        It truncates the inputs to the maximum allowed length.

        It also adds all required special tokens: format is [BOS] input [EOS].

        Finally, it is responsible for padding to maximum length in batch and conversion to torch.Tensor.

        Args:
            inputs: A list of tokenized examples from the current batch.

        Returns:
            input_ids for encoder, attention_mask for encoder
        """
        if are_messages:
            bos_token_id = self.msg_bos_token_id
            eos_token_id = self.msg_eos_token_id
            pad_token_id = self.msg_pad_token_id
        else:
            bos_token_id = self.diff_bos_token_id
            eos_token_id = self.diff_eos_token_id
            pad_token_id = self.diff_pad_token_id

        inputs = [[bos_token_id] + example[: self.encoder_context_max_len - 2] + [eos_token_id] for example in inputs]
        inputs_tensors = [torch.tensor(ids, dtype=torch.int64) for ids in inputs]
        masks_tensors = [torch.ones_like(ids) for ids in inputs_tensors]

        # pad tensors to max length in batch
        inputs_max_len = max(len(tensor) for tensor in inputs)
        inputs_tensors = [
            self._pad_tensor(
                tensor,
                pad_len=inputs_max_len - tensor.numel(),
                value=pad_token_id,
                left=False,
            )
            for tensor in inputs_tensors
        ]
        masks_tensors = [
            self._pad_tensor(
                tensor,
                pad_len=inputs_max_len - tensor.numel(),
                value=0,
                left=False,
            )
            for tensor in masks_tensors
        ]
        return torch.stack(inputs_tensors), torch.stack(masks_tensors)

    def _process_encoder_input(
        self, examples: List[SingleExample]
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
    ]:
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
            results = self._process_inputs(diff_inputs)
        elif self.encoder_input_type == "history":
            history_inputs: List[List[List[int]]] = [example.history_input_ids for example in examples]
            results = self._process_history(history_inputs)
        else:
            raise ValueError("Unknown encoder input type")

        if self.process_retrieved:
            if all(
                example.retrieved_msg_input_ids is not None and example.retrieved_diff_input_ids is not None
                for example in examples
            ):
                retrieved_diff_inputs: List[List[int]] = [example.retrieved_diff_input_ids for example in examples]  # type: ignore[misc]
                retrieved_diff_results = self._process_inputs(retrieved_diff_inputs)
                retrieved_msg_input_ids: List[List[int]] = [example.retrieved_msg_input_ids for example in examples]  # type: ignore[misc]
                retrieved_msg_results = self._process_inputs(retrieved_msg_input_ids, are_messages=True)
                return results, retrieved_diff_results, retrieved_msg_results
            else:
                logging.warning("Collator is configured to process retrieved data, " "but it is not present in input.")
        return results, (None, None), (None, None)
