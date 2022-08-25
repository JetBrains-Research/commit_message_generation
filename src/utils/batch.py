from dataclasses import dataclass
from typing import List

import torch


@dataclass
class Batch:
    diff_input_ids: torch.Tensor
    diff_attention_mask: torch.Tensor
    msg_input_ids: torch.Tensor
    msg_attention_mask: torch.Tensor
    msg_labels: torch.Tensor
    msg_targets: List[str]
    msg_prefixes: List[str]
