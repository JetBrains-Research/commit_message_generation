from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaEncoder, RobertaPooler
from torch import nn
import torch


def create_position_ids_from_input_ids(input_ids, padding_idx, eos_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored.

    This is modified from transformers's `modeling_roberta.create_position_ids_from_input_ids`.
    Sequences before and after [SEP] are enumerated separately.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # original:
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    # TODO: did I break something?
    incremental_indices = torch.zeros_like(input_ids)
    end = torch.where(input_ids == eos_idx)[1][1::3]
    mask = input_ids.ne(padding_idx).int()

    for i, ind in enumerate(end):
        incremental_indices[i, :ind + 1] = (torch.cumsum(input_ids[i, :ind + 1].ne(padding_idx), dim=0).type_as(mask) +
                                            past_key_values_length) * mask[i, :ind + 1]
        incremental_indices[i, ind + 1:] = (torch.cumsum(input_ids[i, ind + 1:].ne(padding_idx), dim=0).type_as(mask) +
                                            past_key_values_length) * mask[i, ind + 1:]
    return incremental_indices.long() + padding_idx


class MyRobertaEmbeddings(nn.Module):
    """
    Same as RobertaEmbeddings with a tiny tweak for positional embeddings.
    """

    # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # End copy
        self.padding_idx = config.pad_token_id
        self.eos_idx = 2  # TODO: how to pass eos_token_id here?
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids. Any padded tokens remain padded.
                position_ids = create_position_ids_from_input_ids(
                    input_ids, self.padding_idx, self.eos_idx, past_key_values_length
                ).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


class MyRobertaModel(RobertaModel):
    # Copied from transformers.models.bert.modeling_roberta.RobertaModel.__init__
    # with RobertaEmbeddings->MyRobertaEmbeddings
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = MyRobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.init_weights()


if __name__ == '__main__':
    from transformers import RobertaTokenizer
    model = MyRobertaModel.from_pretrained('microsoft/codebert-base')
    tok = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    tok.add_special_tokens({"additional_special_tokens": ["<empty>"]})
    model.resize_token_embeddings(len(tok))

    prev_examples = ["<empty> <empty> \n Binary files / dev / null and b / art / intro . png differ \n",
                     "telecomm / java / android / telecomm / Connection . java \n public abstract class Connection { "
                     "\n * / \n public static String stateToString ( int state ) { \n switch ( state ) { \n case "
                     "State . NEW : \n <empty> <empty> <empty> <empty> <empty> <empty> \n <empty> <empty> <empty> "
                     "<empty> <empty> <empty> \n return \" NEW \" ; \n case State . RINGING : \n"]

    upd_examples = ["new file \n Binary files / dev / null and b / art / intro . png differ \n",
                    "telecomm / java / android / telecomm / Connection . java \n public abstract class Connection { "
                    "\n * / \n public static String stateToString ( int state ) { \n switch ( state ) { \n case State "
                    ". NEW : \n + case State . INITIALIZING : \n + return \" INITIALIZING \" ; \n return \" NEW \" ; "
                    "\n case State . RINGING : \n"]

    enc = tok(prev_examples, upd_examples, truncation=True, padding=True, return_tensors='pt', add_special_tokens=True)

    print("Input ids")
    print(enc.input_ids)

    print("Positional ids")
    print(create_position_ids_from_input_ids(enc.input_ids, 1, 2, 0))

    print("Trying forward")
    model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)