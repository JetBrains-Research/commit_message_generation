import os
import argparse
import pickle
import pprint
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer

from Config import Config
from dataset_utils.CommitMessageGenerationDataset import CommitMessageGenerationDataset

from models.EncoderDecoder import EncoderDecoder
from models.SimpleLossCompute import SimpleLossCompute
from models.training.train_utils import make_model, run_epoch, print_examples, add_special_tokens_to_config
from models.evaluation.test_utils import save_perplexity_plot
from models.evaluation.analyze import test_commit_message_generation_model


def train(model: EncoderDecoder, train_iter: DataLoader, val_iter: DataLoader, suffix_for_saving: str,
          config: Config) -> Tuple[List[float], List[float]]:
    """
    :param suffix_for_saving:
    :param tokenizer: RobertaTokenizer TODO: is it needed?
    :param model: models to train
    :param train_iter: train data loader
    :param val_iter: validation data loader
    :param config: config of execution
    :return: train and validation perplexities for each epoch
    """
    pad_index = config['PAD_TOKEN_ID']
    criterion = nn.NLLLoss(reduction="sum", ignore_index=pad_index)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['LEARNING_RATE'])

    train_batches_num: int = len(train_iter)
    train_loss_function = SimpleLossCompute(model.generator, criterion, optimizer)
    train_perplexities = []

    val_batches_num = len(val_iter)
    val_loss_function = SimpleLossCompute(model.generator, criterion, None)
    val_perplexities = []

    epochs_num: int = config['MAX_NUM_OF_EPOCHS']
    min_val_perplexity: float = 10
    num_not_decreasing_steps: int = 0
    early_stopping_rounds: int = config['EARLY_STOPPING_ROUNDS']
    for epoch in range(epochs_num):
        if num_not_decreasing_steps == early_stopping_rounds:
            print(f'Training was early stopped on epoch {epoch} with early stopping rounds {early_stopping_rounds}')
            break

        print(f'Epoch {epoch + 1} / {epochs_num}')
        model.train()
        train_perplexity = run_epoch(train_iter,
                                     model, train_loss_function,
                                     train_batches_num, config['BATCH_SIZE'],
                                     print_every=config['PRINT_EVERY_iTH_BATCH'])
        print(f'Train perplexity: {train_perplexity}')
        train_perplexities.append(train_perplexity)

        model.eval()
        with torch.no_grad():
            print_small_example(model)

            print_examples(val_iter, model, bos_token_id=config['BOS_TOKEN_ID'], eos_token_id=config['EOS_TOKEN_ID'],
                           max_len=config['TOKENS_CODE_CHUNK_MAX_LEN'])

            val_perplexity = run_epoch(val_iter,
                                       model, val_loss_function,
                                       val_batches_num, config['VAL_BATCH_SIZE'],
                                       print_every=config['PRINT_EVERY_iTH_BATCH'])
            print(f'Validation perplexity: {val_perplexity}')
            val_perplexities.append(val_perplexity)
            if val_perplexity < min_val_perplexity:
                save_model(model, f'best_on_validation_{suffix_for_saving}', config)
                min_val_perplexity = val_perplexity
                num_not_decreasing_steps = 0
            else:
                num_not_decreasing_steps += 1

        if epoch % config['SAVE_MODEL_EVERY'] == 0:
            save_data_on_checkpoint(model, train_perplexities, val_perplexities, suffix_for_saving, config)

    return train_perplexities, val_perplexities


def print_small_example(model):
    prev = ["Hello world", "mmm a / build . gradle <nl> subprojects { <nl> } <nl> project . ext { <nl> guavaVersion = ' 14 . 0 . 1 ' <nl> nettyVersion = ' 4 . 0 . 9 . Final ' <nl> slf4jVersion = ' 1 . 7 . 5 ' <nl> commonsIoVersion = ' 2 . 4 ' <nl>"]
    upd = ["Goodbye world", "ppp b / build . gradle <nl> subprojects { <nl> } <nl> project . ext { <nl> guavaVersion = ' 15 . 0 ' <nl> nettyVersion = ' 4 . 0 . 9 . Final ' <nl> slf4jVersion = ' 1 . 7 . 5 ' <nl> commonsIoVersion = ' 2 . 4 ' <nl>"]
    trg = ["Change greeting to farewell", "upgraded guava to 15 . 0"]

    tok = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    src_enc = tok(prev, upd, padding=True, truncation=True, return_tensors='pt')
    trg_enc = tok(trg, padding=True, truncation=True, return_tensors='pt')
    src_enc['input_ids'] = src_enc['input_ids'].to('cuda')
    src_enc['attention_mask'] = src_enc['attention_mask'].to('cuda')
    trg_enc['input_ids'] = trg_enc['input_ids'].to('cuda')
    trg_enc['attention_mask'] = trg_enc['attention_mask'].to('cuda')

    print("Batch size:", 2)
    print("Src seq len:", src_enc['input_ids'].shape[1])
    print("Trg seq len:", trg_enc['input_ids'].shape[1])
    print()

    out, fin = model.encode(input_ids=src_enc['input_ids'], attention_mask=src_enc['attention_mask'])
    print("Encode")
    print("encoder_output", out.shape)
    print("encoder_final", fin.shape)
    print()

    trg_emb = model.get_embeddings(input_ids=trg_enc['input_ids'], attention_mask=trg_enc['attention_mask'])
    print("trg_embed", trg_emb.shape)
    print()

    decoder_states, hidden, pre_output_vectors = model.decode(trg_emb, trg_enc['attention_mask'], out,
                                                              fin, src_enc['attention_mask'].unsqueeze(1).to('cuda'))
    print("Decode")
    print("decoder_states", decoder_states.shape)
    print("hidden", hidden.shape)
    print("pre_output", pre_output_vectors.shape)
    print()

    gen = model.generator(pre_output_vectors)
    print("Generate")
    print("generator(pre_output)", gen.shape)

    _, ind = torch.max(gen, dim=2)
    print("Max probs", torch.exp(_))
    print("Trg:", trg[0])
    print("Pred:", tok.decode(ind[0]))

    print("Trg:", trg[1])
    print("Pred:", tok.decode(ind[1]))
    print()

    loss = nn.NLLLoss(reduction="sum", ignore_index=1)
    print("Loss", loss(gen.view(-1, gen.size(-1)), trg_enc['input_ids'].view(-1)))
    print()


def save_model(model: nn.Module, model_suffix: str, config: Config) -> None:
    torch.save(model.state_dict(), os.path.join(config['OUTPUT_PATH'], f'model_state_dict_{model_suffix}.pt'))
    torch.save(model, os.path.join(config['OUTPUT_PATH'], f'model_{model_suffix}.pt'))
    print(f'Model saved {model_suffix}!')


def load_weights_of_best_model_on_validation(model: nn.Module, suffix: str, config: Config) -> None:
    model.load_state_dict(torch.load(os.path.join(config['OUTPUT_PATH'],
                                                  f'model_best_on_validation_{suffix}.pt')))


def save_data_on_checkpoint(model: nn.Module,
                            train_perplexities: List[float], val_perplexities: List[float],
                            suffix: str,
                            config: Config) -> None:
    save_model(model, f'checkpoint_{suffix}', config)
    with open(os.path.join(config['OUTPUT_PATH'], f'train_perplexities_{suffix}.pkl'), 'wb') as train_file:
        pickle.dump(train_perplexities, train_file)
    with open(os.path.join(config['OUTPUT_PATH'], f'val_perplexities_{suffix}.pkl'), 'wb') as val_file:
        pickle.dump(val_perplexities, val_file)


def run_train(train_iter: DataLoader, val_iter: DataLoader,
              suffix_for_saving: str, config: Config) -> EncoderDecoder:
    print("-------Config--------")
    pprint.pprint(config.get_config())
    config.save()

    model: EncoderDecoder = make_model(emb_size=config['WORD_EMBEDDING_SIZE'],
                                       hidden_size_encoder=config['ENCODER_HIDDEN_SIZE'],
                                       hidden_size_decoder=config['DECODER_HIDDEN_SIZE'],
                                       vocab_size=config['VOCAB_SIZE'],
                                       num_layers=config['NUM_LAYERS'],
                                       dropout=config['DROPOUT'],
                                       use_bridge=config['USE_BRIDGE'],
                                       teacher_forcing_ratio=config['TEACHER_FORCING_RATIO'],
                                       config=config)
    print("----Created model----")
    print(model)

    print("----Training----")
    train_perplexities, val_perplexities = train(model, train_iter, val_iter, suffix_for_saving, config)

    print(train_perplexities)
    print(val_perplexities)

    save_data_on_checkpoint(model, train_perplexities, val_perplexities, suffix_for_saving, config)
    save_perplexity_plot([train_perplexities, val_perplexities], ['train', 'validation'],
                         f'loss_{suffix_for_saving}.png', config)
    load_weights_of_best_model_on_validation(model, suffix_for_saving, config)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int)
    parser.add_argument('--val_size', type=int)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--test_size', type=int)
    parser.add_argument('--test', dest='test', default=False, action='store_true')
    args = parser.parse_args()
    train_size = args.train_size
    val_size = args.val_size
    test_size = args.test_size
    num_epoch = args.num_epoch
    test = args.test

    print("Current working directory:", os.getcwd())

    config = Config()
    add_special_tokens_to_config(RobertaTokenizer.from_pretrained('microsoft/codebert-base'), config)
    if num_epoch:
        config._CONFIG['MAX_NUM_OF_EPOCHS'] = num_epoch
    print('\n====STARTING TRAINING OF COMMIT MESSAGE GENERATOR====\n', end='')
    print("--Constructing datasets--")
    train_dataset_commit = CommitMessageGenerationDataset.load_data(os.path.join(config['DATASET_ROOT'], 'train'),
                                                                    config, size=train_size)
    val_dataset_commit = CommitMessageGenerationDataset.load_data(os.path.join(config['DATASET_ROOT'], 'val'),
                                                                  config, size=val_size)
    test_dataset_commit = CommitMessageGenerationDataset.load_data(os.path.join(config['DATASET_ROOT'], 'test'),
                                                                   config, size=test_size)

    train_loader = DataLoader(train_dataset_commit, batch_size=config['BATCH_SIZE'])
    val_loader = DataLoader(val_dataset_commit, batch_size=config['VAL_BATCH_SIZE'])

    print("Train:", len(train_dataset_commit))
    print("Val:", len(val_dataset_commit))
    print("Test:", len(test_dataset_commit))

    commit_message_generator = run_train(train_loader, val_loader,
                                         'commit_msg_generator', config=config)

    if test:
        print('\n====STARTING EVALUATION OF COMMIT MESSAGE GENERATOR====\n', end='')
        print('\n====BEAM SEARCH====\n')
        test_commit_message_generation_model(commit_message_generator, train_size, val_size, test_size,
                                             config, greedy=False)
        print('\n====GREEDY====\n')
        test_commit_message_generation_model(commit_message_generator, train_size, val_size, test_size,
                                             config, greedy=True)
    return commit_message_generator


if __name__ == "__main__":
    main()
