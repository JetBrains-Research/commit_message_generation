from dataclasses import dataclass
from typing import List

import torch

from src.utils import BatchRetrieval, SingleExample

from .base_collator_utils import BaseCollatorUtils


@dataclass
class DataCollatorRetrieval(BaseCollatorUtils):
    def __call__(self, examples: List[SingleExample]):
        if not self.testing:
            (encoder_input_ids, encoder_attention_mask), _, _ = self._process_encoder_input(examples=examples)
            return BatchRetrieval(
                encoder_input_ids=encoder_input_ids,
                encoder_attention_mask=encoder_attention_mask,
                pos_in_file=[example.pos_in_file for example in examples],
            )
        else:
            batch_size = len(examples)
            return BatchRetrieval(
                encoder_input_ids=torch.randint(1000, (batch_size, self.encoder_context_max_len), dtype=torch.int64),
                encoder_attention_mask=torch.ones(batch_size, self.encoder_context_max_len, dtype=torch.int64),
                pos_in_file=[i for i in range(batch_size)],
            )
