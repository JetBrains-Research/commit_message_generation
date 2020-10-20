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
        encoder_output, encoder_final = self.encode(batch)
        return self.decode(batch['target'], encoder_output, encoder_final, src_mask=batch['attention_mask'])

    def encode(self, batch) -> Tuple[Tensor, Tensor]:
        """
        Encodes prev and updated sequences.

        :param batch: batch to process

        :return: Tuple[[batch_size, sequence_length, hidden_size_encoder],
         [num_layers, batch_size, hidden_size_encoder]]

        Returns a tuple of torch.FloatTensor:

        * encoder_output (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size_encoder)) – Sequence of
        hidden-states at the output of the last layer of the models.

        * encoder_final (torch.FloatTensor of shape (num_layers, batch_size, hidden_size_encoder)

        From RobertaModel.forward() we get Tuple of torch.FloatTensor (one for the output of the embeddings + one for
        the output of each layer) of shape (batch_size, sequence_length, hidden_size_encoder) --- hidden-states of
        the model at the output of each layer plus the initial embedding outputs.

        Then convert it to torch.FloatTensor of shape (num_layers, batch_size, sequence_length, hidden_size_encoder).

        Then take hidden states for t = sequence_length: torch.FloatTensor of shape
        (num_layers, batch_size, hidden_size).
        """
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        encoder_output, _, encoder_final = self.encoder(input_ids, attention_mask=attention_mask,
                                                        output_hidden_states=True)
        encoder_output = encoder_output.to(self.device)
        encoder_final = torch.stack(encoder_final)[:, :, 511, :].to(self.device)  # TODO: pass sequence_length

        return encoder_output, encoder_final

    def decode(self, batch, encoder_output, encoder_final, src_mask, decoder_hidden=None):
        src_mask = src_mask.to(self.device)
        batch['input_ids'] = batch['input_ids'].to(self.device)
        batch['attention_mask'] = batch['attention_mask'].to(self.device)
        return self.decoder(batch, encoder_output, encoder_final,
                            src_mask, hidden=decoder_hidden)
