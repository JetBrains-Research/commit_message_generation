import random
import torch
from torch import nn


class Decoder(nn.Module):
    """A conditional GRU decoder with multihead self-attention."""
    def __init__(self,
                 embed_dim: int,
                 vocab_size: int,
                 hidden_size: int,
                 hidden_size_encoder: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 bridge: bool,
                 teacher_forcing_ratio: float):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.hidden_size_encoder = hidden_size_encoder
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.bridge = nn.Linear(hidden_size_encoder, hidden_size, bias=True) if bridge else None
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.rnn = nn.GRU(hidden_size_encoder + embed_dim,
                          hidden_size,
                          num_layers=num_layers,
                          batch_first=True, dropout=dropout)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size_encoder + hidden_size + embed_dim, hidden_size, bias=False)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward_step(self, prev_embed, encoder_output, src_mask, hidden):
        """Perform a single decoder step (1 word)
        :param prev_embed: embedding of previous target tokens
        (tensor of shape [batch_size, #step, embed_dim])
        :param encoder_output: sequence of hidden-states at the output of the last layer of the encoder
        (tensor of shape [batch_size, src_sequence_length, hidden_size_encoder])
        :param src_mask: attention mask for encoder_output with 0 for pad tokens and 1 for all others
        (tensor of shape [batch_size, src_sequence_length])
        :param proj_key: [batch_size, sequence_length, hidden_size_decoder]
        :param hidden: [batch_size, hidden_size_decoder]
        :return: Tuple[[batch_size, 1, hidden_size_decoder],
                       [num_layers, batch_size, hidden_size_decoder],
                       [batch_size, 1, hidden_size_decoder]]"""


        context, attn_w = self.attention(prev_embed.reshape(prev_embed.shape[1], prev_embed.shape[0], -1),
                                         encoder_output,
                                         encoder_output,
                                         key_padding_mask=src_mask)
        context = context.reshape(context.shape[1], context.shape[0], -1)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        rnn_output, hidden = self.rnn(rnn_input, hidden)

        pre_output = torch.cat([prev_embed, rnn_output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)
        output = self.output_layer(pre_output)
        output = self.log_softmax(output)

        return rnn_output, hidden, output

    def forward(self, input_ids, attention_mask, encoder_output, encoder_final,
                src_attention_mask, hidden=None, max_len=None):
        """
        Unroll the decoder one step at a time.
        :param input_ids: [batch_size, target_sequence_length, hidden_size_encoder]
        :param attention_mask: [batch_size, target_sequence_length]
        :param encoder_output: [batch_size, sequence_length, hidden_size_encoder]
        :param encoder_final: [num_layers, batch_size, hidden_size_encoder]
        :param src_attention_mask: [batch_size, sequence_length]
        :param hidden: decoder hidden state
        :param max_len: the maximum number of steps to unroll the RNN
        :return: decoder_states: [batch_size, target_sequence_length, hidden_size_decoder],
                 hidden: [num_layers, batch_size, hidden_size_decoder],
                 output: [batch_size, target_sequence_length, vocab_size]]
        """

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = input_ids.size(1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)  # [num_layers, batch_size, hidden_size_decoder]

        trg_embed = self.embedding(input_ids)

        encoder_output = encoder_output.reshape(encoder_output.shape[1], encoder_output.shape[0], -1)

        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        output_vectors = []

        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            # first step is always <s> - sos token
            if use_teacher_forcing or i == 0:
                prev_embed = trg_embed[:, i].unsqueeze(1)  # [batch_size, 1, embed_dim]
            else:
                _, prev_pred = torch.max(output, dim=2)
                prev_embed = self.embedding(prev_pred)
            rnn_output, hidden, output = self.forward_step(prev_embed, encoder_output, src_attention_mask, hidden)
            decoder_states.append(rnn_output)
            output_vectors.append(output)

        decoder_states = torch.cat(decoder_states, dim=1)
        output_vectors = torch.cat(output_vectors, dim=1)
        return decoder_states, hidden, output_vectors

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""
        if encoder_final is None:
            return None  # start with zeros
        return torch.tanh(self.bridge(encoder_final))


if __name__ == "__main__":
    from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
    import numpy as np
    print("Just a small run with random examples\n")

    prev = ["mmm a / Wiki . txt <nl> Test auto push 9 . <nl> public void asyncHtml ( ) { <nl>",
            "mmm a / build . gradle <nl> subprojects { <nl> } <nl> project . ext { <nl> guavaVersion = ' 14 . 0 . 1 ' "
            "<nl> nettyVersion = ' 4 . 0 . 9 . Final ' <nl> slf4jVersion = ' 1 . 7 . 5 ' <nl> commonsIoVersion = ' 2 "
            ". 4 ' <nl>"]
    upd = ["ppp b / Wiki . txt <nl> Test auto push 10 . <nl> public void asyncHtml ( ) { <nl>",
           "ppp b / build . gradle <nl> subprojects { <nl> } <nl> project . ext { <nl> guavaVersion = ' 15 . 0 ' <nl> "
           "nettyVersion = ' 4 . 0 . 9 . Final ' <nl> slf4jVersion = ' 1 . 7 . 5 ' <nl> commonsIoVersion = ' 2 . 4 ' "
           "<nl>"]
    trg = ["dddd Please enter the commit message for your changes . ", "upgraded guava to 15 . 0"]

    tok = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    encoder_config = RobertaConfig.from_pretrained('microsoft/codebert-base')
    encoder = RobertaModel.from_pretrained('microsoft/codebert-base', config=encoder_config)

    src_enc = tok(prev, upd, padding=True, truncation=True, return_tensors='pt')
    trg_enc = tok(trg, padding=True, truncation=True, return_tensors='pt')

    print("Batch size:", 2)
    print("Src seq len:", src_enc['input_ids'].shape[1])
    print("Trg seq len:", trg_enc['input_ids'].shape[1])
    print()

    decoder = Decoder(embed_dim=768,
                      vocab_size=tok.vocab_size,
                      hidden_size=256,
                      hidden_size_encoder=768,
                      num_heads=8,
                      num_layers=2,
                      dropout=0.2,
                      bridge=True,
                      teacher_forcing_ratio=-1.0)

    encoder_output, _, encoder_final = encoder(input_ids=src_enc['input_ids'],
                                               attention_mask=src_enc['attention_mask'],
                                               output_hidden_states=True)
    t = encoder_final[0].shape[1] - 1
    encoder_final = torch.stack(encoder_final)[:, :, t, :][-decoder.num_layers:, :]
    print("Encode")
    print("encoder_output", encoder_output.shape)
    print("encoder_final", encoder_final.shape)
    print()

    decoder_states, hidden, output = decoder(input_ids=trg_enc['input_ids'],
                                             attention_mask=trg_enc['attention_mask'],
                                             encoder_output=encoder_output,
                                             encoder_final=encoder_final,
                                             src_attention_mask=torch.logical_not(src_enc['attention_mask']))
    print("Decode")
    print("decoder_states", decoder_states.shape)
    print("hidden", hidden.shape)
    print("output", output.shape)
    print()

    probs, top = torch.max(output, dim=2)
    print("Trg:", tok.decode(trg_enc['input_ids'][0], clean_up_tokenization_spaces=False))
    print("Pred:", tok.decode(top[0], clean_up_tokenization_spaces=False))
    print("Probs:", torch.exp(probs[0]))
    print()
    print("Trg:", tok.decode(trg_enc['input_ids'][1], clean_up_tokenization_spaces=False))
    print("Pred:", tok.decode(top[1], clean_up_tokenization_spaces=False))
    print("Probs:", torch.exp(probs[1]))
    print()

    print(top)
    print()
    print([tok.decode(lst, skip_special_tokens=True, clean_up_tokenization_spaces=False).split(' ') for lst in trg_enc['input_ids'].tolist()])