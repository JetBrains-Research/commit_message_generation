from typing import Tuple

import torch
from torch import nn, Tensor

from models import Decoder
from models import Generator
from transformers import RobertaModel


class EncoderDecoder(nn.Module):
    # TODO: deal with layers dimensions properly in decoder, attention and generator
    """
    Model with Encoder-Decoder architecture.

    Encoder: RobertaModel (CodeBERT)

    Decoder: GRU with attention mechanism
    """

    def __init__(self, encoder: RobertaModel, decoder: Decoder, generator: Generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, batch):
        """Process masked src and target sequences."""
        encoder_hidden, encoder_final = self.encode(batch)
        return self.decode(batch['target'], encoder_hidden, encoder_final, src_mask=batch['attention_mask'])

    def encode(self, batch) -> Tuple[Tensor, Tensor]:
        """
        Encodes prev and updated sequences

        :param batch: batch to process

        :return: a tuple of torch.FloatTensor

        encoder_hidden (tuple(torch.FloatTensor) - Tuple of torch.FloatTensor (one for the output of the embeddings +
        one for the output of each layer) of shape (batch_size, sequence_length, hidden_size). Hidden-states of the
        models at the output of each layer plus the initial embedding outputs.

        encoder_final (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)) – Sequence of
        hidden-states at the output of the last layer of the models.
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        # RobertaModel.forward()
        encoder_final, _, encoder_hidden = self.encoder(input_ids, attention_mask=attention_mask,
                                                        output_hidden_states=True)
        # convert tuple to tensor
        encoder_hidden = torch.stack(encoder_hidden)
        return encoder_hidden, encoder_final

    def decode(self, batch, encoder_hidden, encoder_final, src_mask, decoder_hidden=None):
        # decoder.forward()
        return self.decoder(batch, encoder_hidden, encoder_final,
                            src_mask, hidden=decoder_hidden)
