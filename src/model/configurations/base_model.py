from typing import Any

from torch import nn

from src.utils import Batch


class BaseModel(nn.Module):
    def forward(self, batch: Batch) -> Any:
        raise NotImplementedError()

    def generate(self, batch: Batch, **kwargs) -> Any:
        raise NotImplementedError()
