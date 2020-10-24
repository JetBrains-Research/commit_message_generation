from torch import nn
import torch.nn.functional as F


class GeneratorModel(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(GeneratorModel, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        """
        Projects hidden representation to vocabulary size vector and then softmax to probabilities.
        :param x: [batch_size, target_sequence_length, hidden_size_decoder]
        :return: [batch_size, target_sequence_length, vocab_size]
        """
        return F.log_softmax(self.proj(x), dim=-1)
