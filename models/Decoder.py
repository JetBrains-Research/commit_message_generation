import torch
from torch import nn
from models import BahdanauAttention
import random
from typing import Callable


class Decoder(nn.Module):
    """A conditional GRU decoder with attention."""

    def __init__(self, emb_size: int, hidden_size: int, hidden_size_encoder: int, attention: BahdanauAttention,
                 num_layers: int, dropout: float, bridge: bool, teacher_forcing_ratio: float):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.rnn = nn.GRU(hidden_size_encoder + emb_size, hidden_size, num_layers=num_layers,
                          batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        self.bridge = nn.Linear(hidden_size_encoder, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size_encoder + hidden_size + emb_size,
                                          hidden_size, bias=False)

    def forward_step(self, prev_embed, encoder_output, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)

        :param prev_embed: embedding of previous target token
        (tensor of shape [batch_size, 1, hidden_size_encoder])

        :param encoder_output: sequence of hidden-states at the output of the last layer of the encoder
        (tensor of shape [batch_size, sequence_length, hidden_size_encoder])

        :param src_mask: attention mask for encoder_output with 0 for pad tokens and 1 for all others
        (tensor of shape [batch_size, 1, sequence_length])

        :param proj_key: [batch_size, sequence_length, hidden_size_decoder]

        :param hidden: [batch_size, hidden_size_decoder]

        :return: Tuple[[batch_size, 1, hidden_size_decoder],
                       [num_layers, batch_size, hidden_size_decoder],
                       [batch_size, 1, hidden_size_decoder]]"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]

        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_output, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)

        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)
        return output, hidden, pre_output

    def forward(self, trg_embed, trg_mask, encoder_output, encoder_final,
                src_mask, embedding, generator, hidden=None, max_len=None):
        """
        Unroll the decoder one step at a time.
        :param trg_embed: [batch_size, target_sequence_length, hidden_size_encoder]
        :param trg_mask: [batch_size, target_sequence_length]
        :param encoder_output: [batch_size, sequence_length, hidden_size_encoder]
        :param encoder_final: [num_layers, batch_size, hidden_size_encoder]
        :param src_mask: [batch_size, 1, sequence_length]
        :param embedding
        :param generator
        :param hidden: decoder hidden state
        :param max_len: the maximum number of steps to unroll the RNN
        :return: decoder_states: [batch_size, target_sequence_length, hidden_size_decoder],
                 hidden: [num_layers, batch_size, hidden_size_decoder],
                 pre_output: [batch_size, target_sequence_length, hidden_size_decoder]]
                 (pre_output is a concatenation of prev_embed, output from RNN and context from attention)
        """

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)  # [num_layers, batch_size (or just 1), hidden_size_decoder]

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_output)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            if use_teacher_forcing or i == 0:
                prev_embed = trg_embed[:, i].unsqueeze(1)  # [B, 1, EmbCode]
            else:
                _, top_i = generator(pre_output_vectors[-1]).squeeze(1).topk(1)
                top_i_mask = torch.ones_like(top_i)
                prev_embed = embedding(top_i, top_i_mask)
            output, hidden, pre_output = self.forward_step(prev_embed, encoder_output, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""
        if encoder_final is None:
            return None  # start with zeros
        return torch.tanh(self.bridge(encoder_final))            
