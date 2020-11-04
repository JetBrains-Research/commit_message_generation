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


def decode_tokens(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True):
    tok = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    return tok.decode(seq, skip_special_tokens=skip_special_tokens,
                      clean_up_tokenization_spaces=clean_up_tokenization_spaces)


def make_model(emb_size: int,
               hidden_size_encoder: int,
               hidden_size_decoder: int,
               vocab_size: int,
               num_layers: int,
               dropout: float,
               teacher_forcing_ratio,
               use_bridge: bool,
               config: Config) -> EncoderDecoder:
    """Helper function: Construct an EncoderDecoder model from hyperparameters."""

    codebert_config = RobertaConfig.from_pretrained("microsoft/codebert-base", output_hidden_states=True)
    codebert_model = RobertaModel.from_pretrained("microsoft/codebert-base", config=codebert_config)

    attention = BahdanauAttention(hidden_size_decoder, key_size=hidden_size_encoder, query_size=hidden_size_decoder)
    decoder = Decoder(emb_size, hidden_size_decoder, hidden_size_encoder, attention, num_layers, dropout, use_bridge,
                      teacher_forcing_ratio=teacher_forcing_ratio)
    generator = GeneratorModel(hidden_size_decoder, vocab_size)

    model: EncoderDecoder = EncoderDecoder(
        codebert_model,
        decoder,
        generator)
    model.to(config['DEVICE'])
    return model


def run_epoch(data_iter: Generator, model: EncoderDecoder, loss_compute: SimpleLossCompute,
              batches_num: int, print_every: int, logger) -> float:
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
        batch_size = len(batch['target']['input_ids'])

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
            logger.info(f'Epoch Step: {i} / {batches_num} '
                        f'Loss: {loss / batch_size :.2f} '
                        f'Tokens per Sec: {print_tokens / elapsed :.2f}')
            start = time.time()
            print_tokens = 0
    epoch_duration = time.time() - epoch_start
    logger.info(f'Epoch ended with duration {str(timedelta(seconds=epoch_duration))}')
    return math.exp(total_loss / float(total_tokens))


def greedy_decode(model, batch, sos_index, eos_index, max_len):
    """Greedily decode a sentence."""
    with torch.no_grad():
        encoder_output, encoder_final = model.encode(batch['input_ids'], batch['attention_mask'])
        prev_y = torch.ones(len(batch['target']['input_ids']), 1).fill_(sos_index).type_as(batch['input_ids'])
        trg_mask = torch.ones_like(prev_y)
    output = torch.zeros((batch['input_ids'].shape[0], max_len))
    attention_scores = []
    hidden = None

    for i in range(max_len):
        with torch.no_grad():
            trg_embed = model.get_embeddings(prev_y, trg_mask)
            out, hidden, pre_output = model.decode(trg_embed, trg_mask, encoder_output, encoder_final,
                                                   batch['attention_mask'].unsqueeze(1), hidden=hidden)
            # we predict from the pre-output layer, which is
            # a combination of Decoder state, prev emb, and context
            prob = model.generator(pre_output)[:, -1]  # [batch_size, vocab_size]
        _, next_word = torch.max(prob, dim=1)
        output[:, i] = next_word
        prev_y[:, 0] = next_word  # change prev id to generated id
        attention_scores.append(model.decoder.attention.alphas.cpu().numpy())
    output = output.cpu().long().numpy()
    return output, np.concatenate(attention_scores, axis=1)


def print_examples(example_iter: DataLoader, model: EncoderDecoder, bos_token_id: int, eos_token_id: int, logger,
                   n=5, max_len=30) -> None:
    """Prints N examples. Assumes batch size of 1."""
    count = 0

    for i, batch in enumerate(example_iter):
        batch['input_ids'] = batch['input_ids'].to(model.device)
        batch['attention_mask'] = batch['attention_mask'].to(model.device)
        batch['target']['input_ids'] = batch['target']['input_ids'].to(model.device)
        batch['target']['attention_mask'] = batch['target']['attention_mask'].to(model.device)

        src = batch['input_ids'].cpu().numpy()[0, :]
        trg = batch['target']['input_ids'].cpu().numpy()[0, :]

        result, _ = greedy_decode(model, batch, sos_index=bos_token_id, eos_index=eos_token_id,
                                  max_len=max_len)

        logger.info("Example #%d" % (i + 1))
        logger.info("Src : ", decode_tokens(src, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        logger.info("Trg : ", decode_tokens(trg, skip_special_tokens=True, clean_up_tokenization_spaces=False))
        logger.info("Pred: ", decode_tokens(result[0, :], skip_special_tokens=True, clean_up_tokenization_spaces=False))
        logger.info()

        count += 1
        if count == n:
            break


def create_greedy_decode_method_with_batch_support(model: EncoderDecoder, max_len: int, sos_index: int,
                                                   eos_index: int):
    def decode(batch) -> List[List[np.array]]:
        predicted, _ = greedy_decode(model, batch, sos_index, eos_index, max_len)
        # TODO: no batch support for now?
        return predicted.reshape(predicted.shape[0], 1, -1)  # [batch_size, k, pred_seq_len]

    return decode


def calculate_accuracy(dataset_iterator: Iterable,
                       model: EncoderDecoder,
                       max_len: int,
                       bos_token_id: int,
                       eos_token_id: int) -> float:
    correct = 0
    total = 0
    for batch in dataset_iterator:
        batch['input_ids'] = batch['input_ids'].to('cuda')
        batch['attention_mask'] = batch['attention_mask'].to('cuda')
        batch['target']['input_ids'] = batch['target']['input_ids'].to('cuda')
        batch['target']['attention_mask'] = batch['target']['attention_mask'].to('cuda')
        targets = batch['target']['input_ids']
        results = greedy_decode(model, batch, bos_token_id, eos_token_id, max_len)
        for i in range(len(targets)):
            if np.all(targets[i] == results[i]):
                correct += 1
            total += 1
    return correct / total


def calculate_top_k_accuracy(topk_values: List[int], dataset_iterator: Iterator, decode_method) \
        -> Tuple[List[int], int, List[List[str]]]:
    correct = [0 for _ in range(len(topk_values))]
    total = 0
    max_top_k_decoded = []
    for batch in dataset_iterator:
        batch['input_ids'] = batch['input_ids'].to('cuda')
        batch['attention_mask'] = batch['attention_mask'].to('cuda')
        batch['target']['input_ids'] = batch['target']['input_ids'].to('cuda')
        batch['target']['attention_mask'] = batch['target']['attention_mask'].to('cuda')
        targets = batch['target']['input_ids']  # [batch_size, trg_seq_len]
        results = decode_method(batch)  # [batch_size, k, trg_seq_len])
        for example_id in range(len(targets)):
            target = decode_tokens(targets[example_id], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=False).split()  # [trg_seq_len]
            example_top_k_results = results[example_id][:max_k]  # [max_k, trg_seq_len]
            decoded_tokens = [[decode_tokens(result, skip_special_tokens=True, clean_up_tokenization_spaces=False)]
                              for result in example_top_k_results]
            max_top_k_decoded.append(decoded_tokens)
            tail_id = 0
            print("trg:", target)
            for i, result in enumerate(example_top_k_results):
                result = decode_tokens(result, skip_special_tokens=True, clean_up_tokenization_spaces=False).split()
                print("pred:", result)
                if i + 1 > topk_values[tail_id]:
                    tail_id += 1
                if len(result) == len(target) and np.all(result == target):
                    for j in range(tail_id, len(correct)):
                        correct[j] += 1
                    break
        total += len(batch['target']['input_ids'])
    return correct, total, max_top_k_decoded


def add_special_tokens_to_config(tokenizer: RobertaTokenizer, config: Config):
    config._CONFIG['PAD_TOKEN_ID'] = tokenizer.pad_token_id
    config._CONFIG['EOS_TOKEN_ID'] = tokenizer.eos_token_id
    config._CONFIG['BOS_TOKEN_ID'] = tokenizer.bos_token_id
    config._CONFIG['VOCAB_SIZE'] = tokenizer.vocab_size
