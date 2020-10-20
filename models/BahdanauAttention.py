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
        :param query: [batch_size, 1, hidden_size_decoder]
        :param proj_key: [batch_size, sequence_length, hidden_size_decoder]
        :param value: [batch_size, sequence_length, hidden_size_encoder]
        :param mask: [batch_size, 1, sequence_length] // Actual: [1, sequence_length]

        :return: Tuple[[batch_size, 1, hidden_size_encoder], [batch_size, 1, sequence_length]]
        """
        assert mask is not None, "mask is required"
        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computed.

        query = self.query_layer(query)  # [batch_size, 1, hidden_size_decoder]

        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))  # [batch_size, sequence_length, 1]
        scores = scores.squeeze(2).unsqueeze(1)  # [batch_size, 1, sequence_length]

        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))

        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas  # [batch_size, 1, sequence_length]

        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)  # [batch_size, 1, hidden_size_encoder]
        return context, alphas
