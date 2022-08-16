from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollator:
    """This class is used to construct batches out of lists of examples.

    - Commit diffs are simply padded to maximum length.

      Format: `[DIFF_BOS] diff [DIFF_EOS]`

    - For training and validation with Next Token Prediction metrics, there is an option to concatenate each
      commit message with corresponding history.

      Format with history: `[BOS] history_1 [SEP] ... history_k [SEP] message [SEP]`

      Format without history: `[BOS] message [SEP]`

    - For generation, there are options to add message history to decoder context and also add X% of characters of
      each message to decoder context.

      Format with history: `[BOS] history_1 [SEP] ... history_k [SEP] X% characters of message`

      Format without history: `[BOS] X% characters of message`

    Important: assumes that both diffs and messages do not contain special tokens ([BOS]/[EOS]/etc.).

    Args:
        diff_tokenizer: Tokenizer used to tokenize diff.
        msg_tokenizer: Tokenizer used to tokenize messages.
        encoder_context_max_len: Maximum allowed number of tokens in encoder context.
        decoder_context_max_len: Maximum allowed number of tokens in decoder context.
        decoder_sep_tokens: A list of tokens used as [SEP] in decoder context (added between history examples,
         if they are present, and at the end of message).
        with_history: True to add history to decoder context and False otherwise.
        generation: True to construct batches for generation and False otherwise.
        context_ratio: Useful only when generation == True. (context_ratio)% of characters of each message will
         be added to decoder context (should be a float between 0.0 and 1.0).
        testing: True to generate tensors of maximum possible shape with random numbers instead of actually processing
         input data  (used to quickly test whether current batch size fits in GPU memory).
    """

    diff_tokenizer: PreTrainedTokenizerBase
    msg_tokenizer: PreTrainedTokenizerBase
    encoder_context_max_len: int
    decoder_context_max_len: int
    decoder_sep_tokens: List[int]
    with_history: bool
    generation: bool
    context_ratio: Optional[float] = None
    testing: bool = False

    def _get_prefix(
        self, message_ids: List[int], context_len: Optional[int] = None
    ) -> Dict[str, Union[List[int], str]]:
        """Builds context and target for completion-style generation.
        The last word in context is treated as prefix for the first generated word.

        Args:
            message_ids: Input message, tokenized.

        Returns:
        Dictionary with the following keys:
            - msg_input_ids: Context for the model, tokenized.
            - msg_target: Target, string.
            - msg_prefix: Prefix, string.
        """
        message = self.msg_tokenizer.decode(message_ids, skip_special_tokens=True)
        if not context_len:
            context_len = len(message) * self.context_ratio
        input, target = message[: int(context_len)], message[int(context_len) :]

        # if context is empty, use the whole message as target
        # (might happen with very short messages and small context_ratio)
        if not input:
            return {"msg_input_ids": [], "msg_target": target, "msg_prefix": ""}

        # if the last word in context is full, do not use prefix
        if input[-1].isspace() or target[0].isspace():
            context = input
            prefix = ""
        else:
            context, prefix = " ".join(input.split()[:-1]), input.split()[-1]

            if len(context) > 0:
                prefix = " " + prefix

        return {
            "msg_input_ids": self.msg_tokenizer(context, add_special_tokens=False).input_ids,
            "msg_target": target,
            "msg_prefix": prefix,
        }

    def _process_msg_gen(self, message_ids: List[int]) -> Dict[str, Union[List[int], str]]:
        """Builds context and target for generation.
        The last word in context is treated as prefix for the first generated word.

        Args:
            message_ids: Input message, tokenized.

        Returns:
        Dictionary with the following keys:
            - msg_input_ids: Context for the model, tokenized.
            - msg_target: Target, string.
            - msg_prefix: Prefix, string.
        """
        # context_ratio = 0.0 => do not include anything in context
        if self.context_ratio == 0.0:
            return {
                "msg_input_ids": [],
                "msg_target": self.msg_tokenizer.decode(message_ids, skip_special_tokens=True),
                "msg_prefix": "",
            }

        # context_ratio = 1.0 => include the whole message in context
        if self.context_ratio == 1.0:
            return {"msg_input_ids": message_ids, "msg_target": "", "msg_prefix": ""}

        # 0.0 < context_ratio < 1.0 => include X% characters in context and deal with prefix
        return self._get_prefix(message_ids)

    def _pad_tensor(self, input_tensor: torch.Tensor, pad_len: int, value: int, left: bool) -> torch.Tensor:
        return torch.nn.functional.pad(
            input_tensor, pad=[pad_len, 0] if left else [0, pad_len], mode="constant", value=value
        )

    def __call__(self, examples: List[Dict[str, List]]) -> Dict[str, torch.Tensor]:
        if not self.testing:
            diff_inputs: List[List[int]] = [e["diff_input_ids"] for e in examples]
            message_inputs: List[List[int]] = [e["msg_input_ids"] for e in examples]
            history_inputs: List[List[List[int]]] = [e["history_input_ids"] for e in examples]

            all_diff_ids = [
                [self.diff_tokenizer.bos_token_id]
                + diff[: self.decoder_context_max_len - 2]
                + [self.diff_tokenizer.eos_token_id]
                for diff in diff_inputs
            ]

            all_msg_ids = []  # input for training or NTP metrics: history + cur_msg (right-side padding)
            all_msg_masks = []  # 0 on pad tokens and 1 otherwise (right-side padding)
            all_msg_labels = []  # -100 on history & padding to avoid computing loss (right-side padding)
            all_msg_targets = []
            all_msg_prefixes = []

            for message_ids, history_ids in zip(message_inputs, history_inputs):
                message_ids = message_ids[
                    : self.decoder_context_max_len - 1 - (0 if self.generation else len(self.decoder_sep_tokens))
                ]

                if self.generation:
                    assert self.context_ratio is not None

                    results = self._process_msg_gen(message_ids)
                    message_ids = results["msg_input_ids"]
                    target = results["msg_target"]
                    prefix = results["msg_prefix"]
                else:
                    target = ""
                    prefix = ""

                cur_history_ids = []
                cur_history_labels = []

                # insert previous messages from history until we reach max_len
                if self.with_history:
                    cur_len = len(message_ids) + 1 + (0 if self.generation else len(self.decoder_sep_tokens))
                    for history_input_ids in history_ids[::-1]:
                        if (
                            cur_len + len(history_input_ids) + len(self.decoder_sep_tokens)
                            > self.decoder_context_max_len
                        ):
                            break

                        cur_len += len(history_input_ids) + len(self.decoder_sep_tokens)
                        cur_history_ids.append(history_input_ids + self.decoder_sep_tokens)
                        cur_history_labels.append([-100 for _ in history_input_ids + self.decoder_sep_tokens])

                cur_ids = [[self.msg_tokenizer.bos_token_id]]
                cur_ids.extend(cur_history_ids)
                cur_ids.append(message_ids)

                cur_labels = [[-100]]
                cur_labels.extend(cur_history_labels)
                cur_labels.append(message_ids)

                # do not add [SEP] tokens at the end for generation prompt
                if not self.generation:
                    cur_ids.append(self.decoder_sep_tokens)
                    cur_labels.append(self.decoder_sep_tokens)

                cur_ids = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)
                cur_labels = torch.tensor([ex for sublist in cur_labels for ex in sublist], dtype=torch.int64)
                cur_mask = torch.ones_like(cur_ids)

                all_msg_ids.append(cur_ids)
                all_msg_masks.append(cur_mask)
                all_msg_labels.append(cur_labels)
                all_msg_targets.append(target)
                all_msg_prefixes.append(prefix)

            all_diff_ids = [torch.tensor(ids, dtype=torch.int64) for ids in all_diff_ids]
            all_diff_masks = [torch.ones_like(ids) for ids in all_diff_ids]

            msg_max_len = max(len(tensor) for tensor in all_msg_ids)
            diff_max_len = max(len(tensor) for tensor in all_diff_ids)

            # pad tensors to max length in batch
            for i in range(len(all_diff_ids)):
                all_diff_ids[i] = self._pad_tensor(
                    all_diff_ids[i],
                    pad_len=diff_max_len - all_diff_ids[i].numel(),
                    value=self.diff_tokenizer.pad_token_id,
                    left=False,
                )
                all_diff_masks[i] = self._pad_tensor(
                    all_diff_masks[i], pad_len=diff_max_len - all_diff_masks[i].numel(), value=0, left=False
                )

                # NOTE: left side padding on generation!! https://github.com/huggingface/transformers/issues/3021
                all_msg_ids[i] = self._pad_tensor(
                    all_msg_ids[i],
                    pad_len=msg_max_len - all_msg_ids[i].numel(),
                    value=self.msg_tokenizer.pad_token_id,
                    left=self.generation,
                )
                all_msg_labels[i] = self._pad_tensor(
                    all_msg_labels[i], pad_len=msg_max_len - all_msg_labels[i].numel(), value=-100, left=self.generation
                )
                all_msg_masks[i] = self._pad_tensor(
                    all_msg_masks[i], pad_len=msg_max_len - all_msg_masks[i].numel(), value=0, left=self.generation
                )

            all_diff_ids = torch.stack(all_diff_ids)
            all_diff_masks = torch.stack(all_diff_masks)

            all_msg_ids = torch.stack(all_msg_ids)
            all_msg_masks = torch.stack(all_msg_masks)
            all_msg_labels = torch.stack(all_msg_labels)

            batch_result = {
                "diff_input_ids": all_diff_ids,
                "diff_attention_mask": all_diff_masks,
                "msg_input_ids": all_msg_ids,
                "msg_attention_mask": all_msg_masks,
            }
            if self.generation:
                batch_result.update(
                    {
                        "msg_target": all_msg_targets,
                        "msg_prefix": all_msg_prefixes,
                    }
                )
            else:
                batch_result.update({"msg_labels": all_msg_labels})

            return batch_result
        else:
            batch_size = len(examples)
            return {
                "diff_input_ids": torch.randint(
                    self.diff_tokenizer.vocab_size, (batch_size, self.encoder_context_max_len), dtype=torch.int64
                ),
                "diff_attention_mask": torch.ones(batch_size, self.encoder_context_max_len, dtype=torch.int64),
                "msg_input_ids": torch.randint(
                    self.msg_tokenizer.vocab_size, (batch_size, self.decoder_context_max_len), dtype=torch.int64
                ),
                "msg_attention_mask": torch.ones(batch_size, self.decoder_context_max_len, dtype=torch.int64),
                "msg_labels": torch.randint(
                    self.msg_tokenizer.vocab_size, (batch_size, self.decoder_context_max_len), dtype=torch.int64
                ),
            }
