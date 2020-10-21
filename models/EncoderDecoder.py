from typing import Tuple

import torch
from torch import nn, Tensor

from models import Decoder
from models import Generator
from transformers import RobertaModel


class EncoderDecoder(nn.Module):
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
        encoder_output, encoder_final = self.encode(batch['input_ids'], batch['attention_mask'])
        # TODO: embeddings from RoBERTa as input to decoder?
        trg_embed = self.get_embeddings(batch['target']['input_ids'], batch['target']['attention_mask'])
        return self.decode(trg_embed=trg_embed,
                           trg_mask=batch['target']['attention_mask'].unsqueeze(1).to(self.device),
                           encoder_output=encoder_output,
                           encoder_final=encoder_final,
                           src_mask=batch['attention_mask'].unsqueeze(1).to(self.device))

    def encode(self, input_ids, attention_mask) -> Tuple[Tensor, Tensor]:
        """
        Encodes prev and updated sequences.

        :param input_ids
        :param attention_mask

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
        encoder_output, _, encoder_final = self.encoder(input_ids=input_ids,
                                                        attention_mask=attention_mask,
                                                        output_hidden_states=True)
        encoder_output = encoder_output
        t = encoder_final[0].shape[1] - 1
        encoder_final = torch.stack(encoder_final)[:, :, t, :]
        return encoder_output, encoder_final

    def get_embeddings(self, input_ids, attention_mask) -> Tuple[Tensor, Tensor]:
        """
        Returns embeddings for input sequence.

        :param input_ids
        :param attention_mask

        :return: [batch_size, sequence_length, hidden_size_encoder]
        """
        output = self.encoder(input_ids, attention_mask=attention_mask)
        return output[-1][-1]  # TODO: double-check that that is indeed RoBERTa embeddings

    def decode(self, trg_embed, trg_mask, encoder_output, encoder_final, src_mask, hidden=None):
        return self.decoder(trg_embed, trg_mask, encoder_output, encoder_final,
                            src_mask, hidden=hidden)
