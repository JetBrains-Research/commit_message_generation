from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerFast

from src.utils import BatchTest, BatchTrain, SingleExample


@dataclass
class BaseCollatorUtils:
    """This class is used to construct batches out of lists of examples for training/validation logic.

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


@dataclass
class DataCollatorTrain(BaseCollatorUtils):
    """This class is used to construct batches out of lists of examples for training/validation logic.

    There is an option to add message history to decoder context
    (but if history is used as encoder input, it will be ignored).

    - Format with history: `[BOS] history_1 [SEP] ... history_k [SEP] message [EOS]`

    - Format without history: `[BOS] message [EOS]`

    Args:
        shift_labels: True to mimic transformers' EncoderDecoderModel ids/labels construction, False otherwise
         (pass False for all other model classes).
    """

    shift_labels: bool

    def _shift_for_encoder_decoder(
        self, ids: List[List[int]], labels: List[List[int]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """This method mimics transformers logic of ids and labels for EncoderDecoderModel.

        Starting from transformers v4.12, loss is now calculated in EncoderDecoderModel, not in decoder class.
        Also, decoder input ids are created automatically based on labels: labels are shifted and -100 is replaced
        with pad token. In our case, history ids are masked -100 in labels, but they are still
        meaningful ids. Therefore, we can't use the default approach.
        """
        ids = [[self.msg_tokenizer.bos_token_id]] + ids[:-1]  # type: ignore[attr-defined]
        return ids, labels

    def _process_decoder_input(self, examples: List[SingleExample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        Args:
            examples:

        Returns:

        """
        message_inputs: List[List[int]] = [example.msg_input_ids for example in examples]
        history_inputs: List[List[List[int]]] = [example.history_input_ids for example in examples]

        all_msg_ids: List[torch.Tensor] = []
        all_msg_masks: List[torch.Tensor] = []
        all_msg_labels: List[torch.Tensor] = []

        for message_ids, history_ids in zip(message_inputs, history_inputs):
            message_ids = message_ids[: self.decoder_context_max_len - 2]

            cur_history_ids = []
            cur_history_labels = []

            if self.encoder_input_type != "history" and self.with_history:
                cur_history_ids = self._get_history(
                    cur_len=len(message_ids) + 2,
                    history_ids=history_ids,
                )
                cur_history_labels = [[-100 for _ in message] for message in cur_history_ids]

            cur_ids = (
                [[self.msg_tokenizer.bos_token_id]]  # type: ignore[attr-defined]
                + cur_history_ids
                + [message_ids]
                + [[self.msg_tokenizer.eos_token_id]]  # type: ignore[attr-defined]
            )
            cur_labels = (
                [[self.msg_tokenizer.bos_token_id]]  # type: ignore[attr-defined]
                + cur_history_labels
                + [message_ids]
                + [[self.msg_tokenizer.eos_token_id]]  # type: ignore[attr-defined]
            )

            if self.shift_labels:
                cur_ids, cur_labels = self._shift_for_encoder_decoder(cur_ids, cur_labels)

            cur_ids_tensor = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)
            cur_labels_tensor = torch.tensor([ex for sublist in cur_labels for ex in sublist], dtype=torch.int64)
            cur_mask_tensor = torch.ones_like(cur_ids_tensor)

            all_msg_ids.append(cur_ids_tensor)
            all_msg_masks.append(cur_mask_tensor)
            all_msg_labels.append(cur_labels_tensor)

        msg_max_len = max(len(tensor) for tensor in all_msg_ids)

        # NOTE: left side padding on generation!! https://github.com/huggingface/transformers/issues/3021
        all_msg_ids = [
            self._pad_tensor(
                tensor,
                pad_len=msg_max_len - tensor.numel(),
                value=self.msg_tokenizer.pad_token_id,  # type: ignore[attr-defined]
                left=False,
            )
            for tensor in all_msg_ids
        ]
        all_msg_masks = [
            self._pad_tensor(
                tensor,
                pad_len=msg_max_len - tensor.numel(),
                value=0,
                left=False,
            )
            for tensor in all_msg_masks
        ]
        all_msg_labels = [
            self._pad_tensor(
                tensor,
                pad_len=msg_max_len - tensor.numel(),
                value=-100,
                left=False,
            )
            for tensor in all_msg_labels
        ]

        return torch.stack(all_msg_ids), torch.stack(all_msg_masks), torch.stack(all_msg_labels)

    def __call__(self, examples: List[SingleExample]) -> BatchTrain:
        # explicit checks - some of these ids are allowed to be 0, so otherwise it might lead to an error
        assert self.diff_tokenizer.bos_token_id is not None  # type: ignore[attr-defined]
        assert self.diff_tokenizer.eos_token_id is not None  # type: ignore[attr-defined]
        assert self.diff_tokenizer.pad_token_id is not None  # type: ignore[attr-defined]

        assert self.msg_tokenizer.bos_token_id is not None  # type: ignore[attr-defined]
        assert self.msg_tokenizer.eos_token_id is not None  # type: ignore[attr-defined]
        assert self.msg_tokenizer.pad_token_id is not None  # type: ignore[attr-defined]
        assert self.msg_tokenizer.sep_token_id is not None  # type: ignore[attr-defined]

        if not self.testing:
            encoder_input_ids, encoder_attention_mask = self._process_encoder_input(examples=examples)
            decoder_input_ids, decoder_attention_mask, labels = self._process_decoder_input(examples=examples)

            return BatchTrain(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
        else:
            batch_size = len(examples)
            return BatchTrain(
                encoder_input_ids=torch.randint(
                    self.diff_tokenizer.vocab_size, (batch_size, self.encoder_context_max_len), dtype=torch.int64  # type: ignore[attr-defined]
                ),
                encoder_attention_mask=torch.ones(batch_size, self.encoder_context_max_len, dtype=torch.int64),
                decoder_input_ids=torch.randint(
                    self.msg_tokenizer.vocab_size, (batch_size, self.decoder_context_max_len), dtype=torch.int64  # type: ignore[attr-defined]
                ),
                decoder_attention_mask=torch.ones(batch_size, self.decoder_context_max_len, dtype=torch.int64),
                labels=torch.randint(
                    self.msg_tokenizer.vocab_size, (batch_size, self.decoder_context_max_len), dtype=torch.int64  # type: ignore[attr-defined]
                ),
            )


@dataclass
class DataCollatorTest(BaseCollatorUtils):
    """This class is used to construct batches out of lists of examples for testing logic.

    There is an option to add message history to decoder context
    (but if history is used as encoder input, it will be ignored).
    Also, we try to mimic completion workflow by adding X% of characters of each message
    to decoder context.

    - Format with history: `[BOS] history_1 [SEP] ... history_k [SEP] X% characters of message`

    - Format without history: `[BOS] X% characters of message`

    Args:
        context_ratio: (context_ratio * 100)% of characters of each message will
         be added to decoder context (should be a float between 0.0 and 1.0).
        max_new_tokens: A maximum number of generated tokens during generation. History is added in a way that
          resulting tensor has 2nd dimension <= `decoder_context_max_len` - `max_new_tokens`
          (but this restriction is not applied to input message, it can still be up to `decoder_context_max_len` long).
    """

    context_ratio: float
    max_new_tokens: int = 15  # TODO: make configurable

    def _process_msg_gen(self, message_ids: List[int], context_len: Optional[int] = None) -> Tuple[List[int], str, str]:
        """Builds context and target for completion-style generation.
        The last word in context is treated as prefix for the first generated word.

        Args:
            message_ids: Input message, tokenized.

        Returns: A tuple of length three, where
          - first element is the model input,
          - second element is the target,
          - third element is the prefix.
        """
        # context_ratio = 0.0 => do not include anything in context
        if self.context_ratio == 0.0:
            return [], self.msg_tokenizer.decode(message_ids, skip_special_tokens=True), ""  # type: ignore[attr-defined]

        # context_ratio = 1.0 => include the whole message in context
        if self.context_ratio == 1.0:
            return message_ids, "", ""

        message = self.msg_tokenizer.decode(message_ids, skip_special_tokens=True)  # type: ignore[attr-defined]
        if not context_len:
            assert self.context_ratio
            context_len = int(len(message) * self.context_ratio)
        input, target = message[:context_len], message[context_len:]

        # if context is empty, use the whole message as target
        # (might happen with very short messages and small context_ratio)
        if not input:
            return [], target, ""

        # if the last word in context is full, do not use prefix
        if input[-1].isspace():
            context = input
            prefix = ""
        else:
            context, prefix = " ".join(input.split()[:-1]), input.split()[-1]

            if len(context) > 0:
                prefix = " " + prefix

        return self.msg_tokenizer(context, add_special_tokens=False).input_ids, target, prefix  # type: ignore[operator]

    def _process_decoder_input(
        self, examples: List[SingleExample]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
        message_inputs: List[List[int]] = [example.msg_input_ids for example in examples]
        history_inputs: List[List[List[int]]] = [example.history_input_ids for example in examples]

        all_msg_ids: List[torch.Tensor] = []
        all_msg_masks: List[torch.Tensor] = []

        all_msg_targets: List[str] = []
        all_msg_prefixes: List[str] = []

        for message_ids, history_ids in zip(message_inputs, history_inputs):
            message_ids = message_ids[: self.decoder_context_max_len - 1]
            cur_len = len(message_ids) + 1 + self.max_new_tokens
            message_ids, target, prefix = self._process_msg_gen(message_ids)

            cur_history_ids = []
            if self.encoder_input_type != "history" and self.with_history:
                cur_history_ids = self._get_history(
                    cur_len=cur_len,
                    history_ids=history_ids,
                )

            cur_ids = [[self.msg_tokenizer.bos_token_id]] + cur_history_ids + [message_ids]  # type: ignore[attr-defined]
            cur_ids_tensor = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)
            cur_mask_tensor = torch.ones_like(cur_ids_tensor)

            all_msg_ids.append(cur_ids_tensor)
            all_msg_masks.append(cur_mask_tensor)

            all_msg_targets.append(target)
            all_msg_prefixes.append(prefix)

        msg_max_len = max(len(tensor) for tensor in all_msg_ids)

        # NOTE: left side padding on generation!! https://github.com/huggingface/transformers/issues/3021
        all_msg_ids = [
            self._pad_tensor(
                tensor,
                pad_len=msg_max_len - tensor.numel(),
                value=self.msg_tokenizer.pad_token_id,  # type: ignore[attr-defined]
                left=True,
            )
            for tensor in all_msg_ids
        ]
        all_msg_masks = [
            self._pad_tensor(
                tensor,
                pad_len=msg_max_len - tensor.numel(),
                value=0,
                left=True,
            )
            for tensor in all_msg_masks
        ]

        return torch.stack(all_msg_ids), torch.stack(all_msg_masks), all_msg_targets, all_msg_prefixes

    def __call__(self, examples: List[SingleExample]) -> BatchTest:
        assert self.diff_tokenizer.bos_token_id is not None  # type: ignore[attr-defined]
        assert self.diff_tokenizer.eos_token_id is not None  # type: ignore[attr-defined]
        assert self.diff_tokenizer.pad_token_id is not None  # type: ignore[attr-defined]

        assert self.msg_tokenizer.bos_token_id is not None  # type: ignore[attr-defined]
        assert self.msg_tokenizer.eos_token_id is not None  # type: ignore[attr-defined]
        assert self.msg_tokenizer.pad_token_id is not None  # type: ignore[attr-defined]
        assert self.msg_tokenizer.sep_token_id is not None  # type: ignore[attr-defined]

        if not self.testing:
            encoder_input_ids, encoder_attention_mask = self._process_encoder_input(examples=examples)
            decoder_input_ids, decoder_attention_mask, targets, prefixes = self._process_decoder_input(
                examples=examples
            )

            return BatchTest(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=None,
                targets=targets,
                prefixes=prefixes,
            )
        else:
            batch_size = len(examples)
            return BatchTest(
                encoder_input_ids=torch.randint(
                    self.diff_tokenizer.vocab_size, (batch_size, self.encoder_context_max_len), dtype=torch.int64  # type: ignore[attr-defined]
                ),
                encoder_attention_mask=torch.ones(batch_size, self.encoder_context_max_len, dtype=torch.int64),
                decoder_input_ids=torch.randint(
                    self.msg_tokenizer.vocab_size, (batch_size, self.decoder_context_max_len), dtype=torch.int64  # type: ignore[attr-defined]
                ),
                decoder_attention_mask=torch.ones(batch_size, self.decoder_context_max_len, dtype=torch.int64),
                labels=None,
                targets=[],
                prefixes=[],
            )
