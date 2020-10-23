import math
import time
from typing import Generator, Iterable, List, Iterator, Tuple
from transformers import RobertaTokenizer
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader

from transformers import RobertaModel, RobertaConfig

from models import SimpleLossCompute
from models.EncoderDecoder import EncoderDecoder
from models.Decoder import Decoder
from models.BahdanauAttention import BahdanauAttention
from models.Generator import GeneratorModel

import Config


def make_model(emb_size: int,
               hidden_size_encoder: int,
               hidden_size_decoder: int,
               num_layers: int,
               dropout: float,
               use_bridge: bool,
               config: Config) -> EncoderDecoder:
    """Helper function: Construct an EncoderDecoder model from hyperparameters."""

    codebert_config = RobertaConfig.from_pretrained("microsoft/codebert-base", output_hidden_states=True)
    codebert_model = RobertaModel.from_pretrained("microsoft/codebert-base", config=codebert_config)
    codebert_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")  # need for vocab size only

    attention = BahdanauAttention(hidden_size_decoder, key_size=hidden_size_encoder, query_size=hidden_size_decoder)
    generator = GeneratorModel(hidden_size_decoder, codebert_tokenizer.vocab_size)
    decoder = Decoder(emb_size, hidden_size_decoder, hidden_size_encoder, attention, num_layers, dropout, use_bridge)

    model: EncoderDecoder = EncoderDecoder(
        codebert_model,
        decoder,
        generator)
    model.to(config['DEVICE'])
    return model


def run_epoch(data_iter: Generator, model: EncoderDecoder, loss_compute: SimpleLossCompute,
              batches_num: int, batch_size: int, print_every: int) -> float:
    """
    1 epoch of training.
    :return: loss per token
    """

    start = time.time()
    total_tokens = 0
    print_tokens = 0
    epoch_start = start
    total_loss = 0

    for i, batch in enumerate(data_iter, 1):
        batch['input_ids'] = batch['input_ids'].to(model.device)
        batch['attention_mask'] = batch['attention_mask'].to(model.device)
        batch['target']['input_ids'] = batch['target']['input_ids'].to(model.device)
        batch['target']['attention_mask'] = batch['target']['attention_mask'].to(model.device)

        out, _, pre_output = model.forward(batch)
        loss = loss_compute(pre_output, batch['target']['input_ids'], batch_size)
        total_loss += loss

        # number of tokens in batch
        # attention mask is 0 for pad tokens and 1 for all others
        total_tokens += torch.sum(batch['attention_mask'])
        print_tokens += torch.sum(batch['attention_mask'])

        if model.training and i % print_every == 0:
            elapsed = time.time() - start
            print(f'Epoch Step: {i} / {batches_num} '
                  f'Loss: {loss / batch_size :.2f} '
                  f'Tokens per Sec: {print_tokens / elapsed :.2f}')
            start = time.time()
            print_tokens = 0
    epoch_duration = time.time() - epoch_start
    print(f'Epoch ended with duration {str(timedelta(seconds=epoch_duration))}')
    return math.exp(total_loss / float(total_tokens))


def greedy_decode(model, batch, tokenizer: RobertaTokenizer, max_len=100):
    """Greedily decode a sentence."""
    sos_index = tokenizer.bos_token_id
    eos_index = tokenizer.eos_token_id
    pad_index = tokenizer.pad_token_id

    with torch.no_grad():
        encoder_output, encoder_final = model.encode(batch['input_ids'], batch['attention_mask'])
        prev_y = torch.ones(1, 1).fill_(sos_index).type_as(batch['input_ids'])
        trg_mask = torch.ones_like(prev_y)

    output = torch.zeros((1, max_len))
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            trg_embed = model.get_embeddings(prev_y, trg_mask)
            out, hidden, pre_output = model.decode(trg_embed, trg_mask, encoder_output, encoder_final,
                                                   batch['attention_mask'].unsqueeze(1), hidden=hidden)
            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output[:, -1])  # [batch_size, vocab_size]
        _, next_words = torch.topk(prob, 2)
        print("top 2 probs", tokenizer.decode(next_words))
        # choose next value if <pad> has max probability
        # TODO: i think normally <pad> shouldn't have max probability :(
        next_words = next_words.squeeze()
        next_words = next_words[:1] if next_words[0] != pad_index else next_words[1:]
        output[:, i] = next_words
        prev_y[:, 0] = next_words

    output = output.cpu().long().numpy()
    print("greedy decode output", output)
    return remove_eos(output, eos_index)


def remove_eos(batch: np.array, eos_index: int):
    result = []
    for sequence in batch:
        eos = np.where(sequence == eos_index)[0]
        if eos.shape[0] > 0:
            sequence = sequence[:eos[0]]
        result.append(sequence)
    return result


def print_examples(example_iter: DataLoader, model: EncoderDecoder, tokenizer: RobertaTokenizer,
                   n=10, max_len=100) -> None:
    """Prints N examples. Assumes batch size of 1."""
    count = 0
    print()

    eos_index = tokenizer.eos_token_id

    for i, batch in enumerate(example_iter):
        batch['input_ids'] = batch['input_ids'].to(model.device)
        batch['attention_mask'] = batch['attention_mask'].to(model.device)
        batch['target']['input_ids'] = batch['target']['input_ids'].to(model.device)
        batch['target']['attention_mask'] = batch['target']['attention_mask'].to(model.device)

        src = batch['input_ids'].cpu().numpy()[0, :]
        trg = batch['target']['input_ids'].cpu().numpy()[0, :]

        # remove <s></s> (if it is there)
        src = src[:-1] if src[-1] == eos_index else src
        trg = trg[:-1] if trg[-1] == eos_index else trg

        result = greedy_decode(model, batch, tokenizer, max_len=max_len)
        print("Greedy decode result", result)

        print("Example #%d" % (i + 1))
        print("Src : ", tokenizer.decode(src, skip_special_tokens=True))
        print("Trg : ", tokenizer.decode(trg, skip_special_tokens=True))
        print("Pred: ", tokenizer.decode(result[0]))
        print()

        count += 1
        if count == n:
            break


def calculate_accuracy(dataset_iterator: Iterable,
                       model: EncoderDecoder,
                       tokenizer: RobertaTokenizer,
                       max_len: int,
                       config: Config) -> float:
    sos_index = tokenizer.bos_token_id
    eos_index = tokenizer.eos_token_id

    correct = 0
    total = 0
    for batch in dataset_iterator:
        targets = remove_eos(batch, eos_index)

        results = greedy_decode(model, batch, tokenizer, max_len)
        for i in range(len(targets)):
            if np.all(targets[i] == results[i]):
                correct += 1
            total += 1
    return correct / total


def calculate_top_k_accuracy(topk_values: List[int], dataset_iterator: Iterator, tokenizer: RobertaTokenizer,
                             decode_method, eos_index) \
        -> Tuple[List[int], int, List[List[str]]]:
    correct = [0 for _ in range(len(topk_values))]
    max_k = topk_values[-1]
    total = 0
    max_top_k_results = []
    for batch in dataset_iterator:
        targets = remove_eos(batch, eos_index)
        results = decode_method(batch)
        for example_id in range(len(results)):
            target = targets[example_id]
            example_top_k_results = results[example_id][:max_k]
            decoded_tokens = [tokenizer.decode(result, skip_special_tokens=True) for result in example_top_k_results]
            max_top_k_results.append(decoded_tokens)
            tail_id = 0
            for i, result in enumerate(example_top_k_results):
                if i + 1 > topk_values[tail_id]:
                    tail_id += 1
                if len(result) == len(target) and np.all(result == target):
                    for j in range(tail_id, len(correct)):
                        correct[j] += 1
                    break
        total += len(batch)
    return correct, total, max_top_k_results