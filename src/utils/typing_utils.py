from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class SingleExample:
    diff_input_ids: List[int]
    msg_input_ids: List[int]
    history_input_ids: List[List[int]]


@dataclass
class Batch:
    encoder_input_ids: torch.Tensor
    encoder_attention_mask: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_attention_mask: torch.Tensor
    labels: Optional[torch.Tensor]


@dataclass
class BatchTrain(Batch):
    labels: torch.Tensor


@dataclass
class BatchTest(Batch):
    targets: List[str]
    prefixes: List[str]
