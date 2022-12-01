from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from src.utils import BatchTest, SingleExample

from .base_collator_utils import BaseCollatorUtils


@dataclass
class DataCollatorTest(BaseCollatorUtils):
    """This class is used to construct batches out of lists of examples in evaluation setting.

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
