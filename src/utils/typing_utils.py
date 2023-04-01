from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class SingleExample:
    diff_input_ids: List[int]
    msg_input_ids: List[int]
    history_input_ids: List[List[int]]
    retrieved_diff_input_ids: Optional[List[int]] = None
    retrieved_msg_input_ids: Optional[List[int]] = None


@dataclass
class Batch:
    encoder_input_ids: torch.Tensor
    encoder_attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]
    retrieved_diff_input_ids: Optional[torch.Tensor]
    retrieved_diff_attention_mask: Optional[torch.Tensor]
    retrieved_msg_input_ids: Optional[torch.Tensor]
    retrieved_msg_attention_mask: Optional[torch.Tensor]

    def pin_memory(self):
        self.encoder_input_ids = self.encoder_input_ids.pin_memory()
        self.encoder_attention_mask = self.encoder_attention_mask.pin_memory()
        self.decoder_input_ids = self.encoder_input_ids.pin_memory()
        self.decoder_attention_mask = self.encoder_attention_mask.pin_memory()
        if self.labels:
            self.labels = self.labels.pin_memory()
        if self.retrieved_diff_input_ids:
            self.retrieved_diff_input_ids = self.retrieved_diff_input_ids.pin_memory()
        if self.retrieved_diff_attention_mask:
            self.retrieved_diff_attention_mask = self.retrieved_diff_attention_mask.pin_memory()
        if self.retrieved_msg_input_ids:
            self.retrieved_msg_input_ids = self.retrieved_msg_input_ids.pin_memory()
        if self.retrieved_msg_attention_mask:
            self.retrieved_msg_attention_mask = self.retrieved_msg_attention_mask.pin_memory()
        return self


@dataclass
class BatchTrain(Batch):
    labels: torch.Tensor


@dataclass
class BatchTest(Batch):
    targets: List[str]
    prefixes: List[str]
