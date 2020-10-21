import os
import sys
import pickle
import pprint
from typing import Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import RobertaTokenizer

from models.EncoderDecoder import EncoderDecoder
from models.SimpleLossCompute import SimpleLossCompute
from Config import Config
from dataset_utils.CommitMessageGenerationDataset import CommitMessageGenerationDataset
from models.train_utils import make_model, run_epoch, print_examples
from models.test_utils import save_perplexity_plot


def train(model: EncoderDecoder, tokenizer: RobertaTokenizer,
          train_iter: DataLoader, val_iter: DataLoader, suffix_for_saving: str,
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
    criterion = nn.NLLLoss(reduction="sum")
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

        print(f'Epoch {epoch} / {epochs_num}')
        model.train()
        train_perplexity = run_epoch(train_iter,
                                     model, train_loss_function,
                                     train_batches_num, config['BATCH_SIZE'],
                                     print_every=config['PRINT_EVERY_iTH_BATCH'])
        print(f'Train perplexity: {train_perplexity}')
        train_perplexities.append(train_perplexity)

        model.eval()
        with torch.no_grad():
            print_examples(val_iter,
                           model, tokenizer, max_len=config['TOKENS_CODE_CHUNK_MAX_LEN'], n=3)

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


def save_model(model: nn.Module, model_suffix: str, config: Config) -> None:
    torch.save(model.state_dict(), os.path.join(config['OUTPUT_PATH'], f'model_state_dict_{model_suffix}.pt'))
    torch.save(model, os.path.join(config['OUTPUT_PATH'], f'model_{model_suffix}.pt'))
    print(f'Model saved {model_suffix}!')


def load_weights_of_best_model_on_validation(model: nn.Module, suffix: str, config: Config) -> None:
    model.load_state_dict(torch.load(os.path.join(config['OUTPUT_PATH'],
                                                  f'best_on_validation_{suffix}.pt')))


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
                                       num_layers=config['NUM_LAYERS'],
                                       dropout=config['DROPOUT'],
                                       use_bridge=config['USE_BRIDGE'],
                                       config=config)
    print("----Created model----")
    print(model)

    print("----Training----")
    # TODO: think how to fix chaotic usage of tokenizer in train-related functions
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    train_perplexities, val_perplexities = train(model, tokenizer, train_iter, val_iter, suffix_for_saving, config)

    print(train_perplexities)
    print(val_perplexities)

    save_data_on_checkpoint(model, train_perplexities, val_perplexities, suffix_for_saving, config)
    save_perplexity_plot([train_perplexities, val_perplexities], ['train', 'validation'],
                         f'loss_{suffix_for_saving}.png', config)
    load_weights_of_best_model_on_validation(model, suffix_for_saving, config)
    return model


def main():
    if len(sys.argv) > 1:
        working = sys.argv[1]
        os.chdir(working)

    print("pwd", os.getcwd())

    config = Config()
    print('\n====STARTING TRAINING OF COMMIT MESSAGE GENERATOR====\n', end='')
    print("--Constructing datasets--")
    train_dataset_commit = CommitMessageGenerationDataset.load_data(os.path.join(config['DATASET_ROOT'], 'train'),
                                                                    config)
    val_dataset_commit = CommitMessageGenerationDataset.load_data(os.path.join(config['DATASET_ROOT'], 'val'),
                                                                  config)
    #test_dataset_commit = CommitMessageGenerationDataset.load_data(os.path.join(config['DATASET_ROOT'], 'test'), config)

    train_loader = DataLoader(train_dataset_commit, batch_size=config['BATCH_SIZE'])
    val_loader = DataLoader(val_dataset_commit, batch_size=config['VAL_BATCH_SIZE'])

    print("Train:", len(train_dataset_commit))
    print("Val:", len(val_dataset_commit))
    #print("Test:", len(test_dataset_commit))

    commit_message_generator = run_train(train_loader, val_loader,
                                         'commit_msg_generator', config=config)
    return commit_message_generator


if __name__ == "__main__":
    main()
