import torch
from torch import nn
from models import BahdanauAttention


class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""

    def __init__(self, emb_size: int, hidden_size: int, attention: BahdanauAttention, num_layers: int, dropout: float,
                 bridge: bool):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout

        self.rnn = nn.GRU(emb_size + 2 * hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)

        # to initialize from the final encoder state
        # TODO: input dimension of this layer should match encoder_hidden.shape[-1]
        self.bridge = nn.Linear(768, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2 * hidden_size + emb_size,
                                          hidden_size, bias=False)

    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""
        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output

    def forward(self, batch, encoder_hidden, encoder_final,
                src_mask, hidden=None, max_len=None):
        """
        Unroll the decoder one step at a time.
        :param batch:
        :param encoder_hidden: [B, SrcSeqLen, NumDirections * SrcEncoderH]
        :param encoder_final: Tuple of [NumLayers, B, NumDirections * SrcEncoderH]
        :param src_mask: [B, 1, SrcSeqLen]
        :param hidden: decoder hidden state
        :param max_len: the maximum number of steps to unroll the RNN
        :return: Tuple[
                 [B, TrgSeqLen, DecoderH],
                 [NumLayers, B, DecoderH],
                 [B, TrgSeqLen, DecoderH]]
        """

        # TODO: is batch['input_ids'] good choice for trg_embed?
        trg_embed = batch['input_ids']
        trg_mask = batch['attention_mask']

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        print(encoder_hidden.shape)
        print(self.attention.key_layer)
        proj_key = self.attention.key_layer(encoder_hidden)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
                prev_embed, encoder_hidden, src_mask, proj_key, hidden)
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
