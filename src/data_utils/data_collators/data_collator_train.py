from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from src.utils import BatchTrain, SingleExample

from .base_collator_utils import BaseCollatorUtils


@dataclass
class DataCollatorTrain(BaseCollatorUtils):
    """This class is used to construct batches out of lists of examples in training/validation setting.

    There is an option to add message history to decoder context
    (but if history is used as encoder input, it will be ignored).

    - Format with history: `[BOS] history_1 [SEP] ... history_k [SEP] message [EOS]`

    - Format without history: `[BOS] message [EOS]`

    Attributes:
        shift_labels: True to mimic transformers' seq2seq models ids/labels construction logic, False otherwise
         (pass False for decoder class).
    """

    shift_labels: bool
    decoder_start_token_id: Optional[int] = None

    def _shift_for_encoder_decoder(
        self, ids: List[List[int]], labels: List[List[int]]
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """This method mimics transformers logic of ids and labels for EncoderDecoderModel
        (or T5ForConditionalGeneration).

        Starting from transformers v4.12, loss is now calculated in EncoderDecoderModel, not in decoder class.
        Also, decoder input ids are created automatically based on labels: labels are shifted and -100 is replaced
        with pad token. In our case, history ids are masked -100 in labels, but they are still
        meaningful ids. Therefore, we can't use the default approach.
        """
        if self.decoder_start_token_id is None:
            ids = [[self.msg_bos_token_id]] + ids[:-1]
        else:
            ids = [[self.decoder_start_token_id]] + ids[:-1]
        return ids, labels

    def _process_decoder_input(self, examples: List[SingleExample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepares decoder input for train/validation:
          * aggregates messages from history when configured accordingly
          * concatenates history with current message
          * constructs labels
          * pads, converts to tensors

        Args:
            examples: A list of inputs for current batch.

        Returns:
            Tuple of three tensors: input ids, attention masks, labels.
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

            cur_ids = [[self.msg_bos_token_id]] + cur_history_ids + [message_ids] + [[self.msg_eos_token_id]]
            cur_labels = [[self.msg_bos_token_id]] + cur_history_labels + [message_ids] + [[self.msg_eos_token_id]]

            if self.shift_labels:
                cur_ids, cur_labels = self._shift_for_encoder_decoder(cur_ids, cur_labels)

            cur_ids_tensor = torch.tensor([ex for sublist in cur_ids for ex in sublist], dtype=torch.int64)
            cur_labels_tensor = torch.tensor([ex for sublist in cur_labels for ex in sublist], dtype=torch.int64)
            cur_mask_tensor = torch.ones_like(cur_ids_tensor)

            all_msg_ids.append(cur_ids_tensor)
            all_msg_masks.append(cur_mask_tensor)
            all_msg_labels.append(cur_labels_tensor)

        msg_max_len = max(len(tensor) for tensor in all_msg_ids)
        all_msg_ids = [
            self._pad_tensor(
                tensor,
                pad_len=msg_max_len - tensor.numel(),
                value=self.msg_pad_token_id,
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
        if not self.testing:
            (
                (encoder_input_ids, encoder_attention_mask),
                (retrieved_diff_input_ids, retrieved_diff_attention_mask),
                (retrieved_msg_input_ids, retrieved_msg_attention_mask),
            ) = self._process_encoder_input(examples=examples)

            decoder_input_ids, decoder_attention_mask, labels = self._process_decoder_input(examples=examples)

            return BatchTrain(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                retrieved_diff_input_ids=retrieved_diff_input_ids,
                retrieved_diff_attention_mask=retrieved_diff_attention_mask,
                retrieved_msg_input_ids=retrieved_msg_input_ids,
                retrieved_msg_attention_mask=retrieved_msg_attention_mask,
            )
        else:
            batch_size = len(examples)
            return BatchTrain(
                encoder_input_ids=torch.randint(1000, (batch_size, self.encoder_context_max_len), dtype=torch.int64),
                encoder_attention_mask=torch.ones(batch_size, self.encoder_context_max_len, dtype=torch.int64),
                decoder_input_ids=torch.randint(1000, (batch_size, self.decoder_context_max_len), dtype=torch.int64),
                decoder_attention_mask=torch.ones(batch_size, self.decoder_context_max_len, dtype=torch.int64),
                labels=torch.randint(1000, (batch_size, self.decoder_context_max_len), dtype=torch.int64),
                retrieved_diff_input_ids=torch.randint(
                    1000, (batch_size, self.encoder_context_max_len), dtype=torch.int64
                ),
                retrieved_diff_attention_mask=torch.ones(batch_size, self.encoder_context_max_len, dtype=torch.int64),
                retrieved_msg_input_ids=torch.randint(
                    1000, (batch_size, self.encoder_context_max_len), dtype=torch.int64
                ),
                retrieved_msg_attention_mask=torch.ones(batch_size, self.encoder_context_max_len, dtype=torch.int64),
            )
