import os
import sys
from typing import Tuple
from random import randint

from torchtext.data import Example, Field, Dataset

from dataset_utils.utils import create_filter_predicate_on_code_and_msg
from Config import Config

from transformers import RobertaTokenizer, BatchEncoding


class CommitMessageGenerationDataset(Dataset):
    """Defines a dataset for commit message generation task as torchtext.data.Dataset"""
    def __init__(self, path: str, config: Config, filter_pred, verbose=False) -> None:
        """Create a TranslationDataset given paths.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        codebert_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

        def codebert_tokenize(tokens):
            batch_enc = codebert_tokenizer(tokens, truncation=True, padding=True)
            return codebert_tokenizer.convert_ids_to_tokens(batch_enc.input_ids)

        src_field: Field = Field(batch_first=True, lower=config['LOWER'], include_lengths=True,
                                 unk_token=config['UNK_TOKEN'], pad_token=config['PAD_TOKEN'],
                                 init_token=config['SOS_TOKEN'],
                                 eos_token=config['EOS_TOKEN'], tokenize=codebert_tokenize)
        trg_field: Field = Field(batch_first=True, lower=config['LOWER_COMMIT_MSG'], include_lengths=True,
                                 unk_token=config['UNK_TOKEN'], pad_token=config['PAD_TOKEN'],
                                 init_token=config['SOS_TOKEN'],
                                 eos_token=config['EOS_TOKEN'], tokenize=codebert_tokenize)

        fields = [('src', src_field), ('trg', trg_field),
                  ('src_batch_enc', Field(sequential=False, use_vocab=False, dtype=BatchEncoding)),
                  ('trg_batch_enc', Field(sequential=False, use_vocab=False, dtype=BatchEncoding)),
                  ('ids', Field(sequential=False, use_vocab=False))]

        examples = []

        with open(os.path.join(path, 'diff.txt'), mode='r', encoding='utf-8') as diff, \
                open(os.path.join(path, 'msg.txt'), mode='r', encoding='utf-8') as msg, \
                open(os.path.join(path, 'prev.txt'), mode='r', encoding='utf-8') as prev, \
                open(os.path.join(path, 'updated.txt'), mode='r', encoding='utf-8') as updated:
            for diff_line, msg_line, prev_line, updated_line in zip(diff, msg, prev, updated):
                diff_line, msg_line, prev_line, updated_line = \
                    diff_line.strip(), msg_line.strip(), prev_line.strip(), updated_line.strip()

                is_correct, error = filter_pred((diff_line.split(' '), msg_line.split(' ')))

                if not is_correct:
                    if verbose:
                        print(f'Incorrect example is seen. Error: {error}', file=sys.stderr)
                    continue

                src_encodings = codebert_tokenizer(prev_line, updated_line, truncation=True,
                                                   padding=True, return_tensors='pt')
                trg_encodings = codebert_tokenizer(msg_line, truncation=True, padding=True, return_tensors='pt')

                examples.append(Example.fromlist(
                    [diff_line, msg_line, src_encodings, trg_encodings, len(examples)], fields))
        super().__init__(examples, fields)

    @staticmethod
    def load_data(verbose: bool, config: Config) -> Tuple[Dataset, Dataset, Dataset, Tuple[Field, Field, Field]]:
        filter_predicate = create_filter_predicate_on_code_and_msg(config['TOKENS_CODE_CHUNK_MAX_LEN'],
                                                                   config['MSG_MAX_LEN'])
        train_data = CommitMessageGenerationDataset(os.path.join(config['DATASET_ROOT'], 'train'),
                                                    config, filter_pred=filter_predicate)

        src_field = train_data.fields['src']
        trg_field = train_data.fields['trg']

        val_data = CommitMessageGenerationDataset(os.path.join(config['DATASET_ROOT'], 'val'),
                                                  config, filter_pred=filter_predicate)
        test_data = CommitMessageGenerationDataset(os.path.join(config['DATASET_ROOT'], 'test'),
                                                   config, filter_pred=filter_predicate)
        src_field.build_vocab(train_data.src, min_freq=config['TOKEN_MIN_FREQ'])
        trg_field.build_vocab(train_data.trg, min_freq=config['TOKEN_MIN_FREQ'])

        if verbose:
            CommitMessageGenerationDataset.print_data_info(train_data, val_data, test_data,
                                                           src_field, trg_field, config)
        return train_data, val_data, test_data, (src_field, trg_field)

    @staticmethod
    def print_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset,
                        src_field: Field, trg_field: Field, config: Config) -> None:
        """ This prints some useful stuff about our datasets. """

        print("Dataset sizes (number of sentence pairs):")
        print('train', len(train_data))
        print('valid', len(valid_data))
        print('test', len(test_data), "\n")

        max_src_len = max(len(example.src) for dataset in (train_data, valid_data, test_data) for example in dataset)
        max_trg_len = max(len(example.trg) for dataset in (train_data, valid_data, test_data) for example in dataset)

        print(f'Max src sequence length in tokens: {max_src_len}')
        print(f'Max trg sequence length in tokens: {max_trg_len}')
        print()

        n = randint(0, len(train_data) - 1)
        print(f"Random ({n}'th) training example:")
        print("src (diff):", " ".join(vars(train_data[n])['src']))
        print("src (prev):", " ".join(vars(train_data[n])['src_prev']))
        print("src (updated):", " ".join(vars(train_data[n])['src_upd']))
        print("trg (msg):", " ".join(vars(train_data[n])['trg']))
        print()

        print("Most common words in src vocabulary:")
        print("\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)]), "\n")
        print()
        print("Most common words in trg vocabulary:")
        print("\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)]), "\n")
        print()

        print("First 10 words in src vocabulary:")
        print("\n".join('%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10])), "\n")
        print()
        print("First 10 words in trg vocabulary:")
        print("\n".join('%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10])), "\n")
        print()

        special_tokens = [config['UNK_TOKEN'], config['PAD_TOKEN'], config['SOS_TOKEN'], config['EOS_TOKEN']]
        print("Special words frequency and ids in src vocabulary: ")
        for special_token in special_tokens:
            print(f"{special_token} {src_field.vocab.freqs[special_token]} {src_field.vocab.stoi[special_token]}")
        print("Special words frequency and ids in trg vocabulary: ")
        for special_token in special_tokens:
            print(f"{special_token} {trg_field.vocab.freqs[special_token]} {trg_field.vocab.stoi[special_token]}")
        print()

        print("Number of words (types) in src vocabulary:", len(src_field.vocab))
        print("Number of words (types) in trg vocabulary:", len(trg_field.vocab))
        print()
