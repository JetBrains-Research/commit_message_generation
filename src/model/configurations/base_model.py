from typing import Any

from torch import nn

from src.utils import Batch, BatchTest


class BaseModel(nn.Module):
    def forward(self, batch: Batch) -> Any:
        raise NotImplementedError()

    def generate(self, batch: BatchTest, **kwargs) -> Any:
        raise NotImplementedError()

    def num_parameters(self, exclude_embeddings: bool):
        raise NotImplementedError()
