import torch
import torch.nn.functional as F
from torch import nn, Tensor

from typing import Tuple


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, key_size: int, query_size: int = None):
        super(BahdanauAttention, self).__init__()

        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        # to store attention scores
        self.alphas = None

    def forward(self, query: Tensor, proj_key: Tensor, value: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param query: [B, 1, DecoderH]
        :param proj_key: [B, SrcSeqLen, DecoderH]
        :param value: [B, SrcSeqLen, NumDirections * SrcEncoderH]
        :param mask: [B, 1, SrcSeqLen]
        :return: Tuple[[B, 1, NumDirections * SrcEncoderH], [B, 1, SrcSeqLen]]
        """
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)  # [B, 1, DecoderH]

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))  # [B, SrcSeqLen, 1]
        scores = scores.squeeze(2).unsqueeze(1)  # [B, 1, SrcSeqLen]
        print(scores.shape)
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas  # [B, 1, SrcSeqLen]

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)  # [B, 1, NumDirections * SrcEncoderH]

        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas
