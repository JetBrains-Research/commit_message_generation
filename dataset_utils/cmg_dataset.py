import os
import sys

import torch
from torch.utils.data import Dataset, Subset
from transformers import RobertaTokenizer, GPT2Tokenizer


def create_filter_predicate_on_code_and_msg(max_length_code, max_length_msg):
    """Create function to check length of diffs and target messages."""

    def filter_predicate(diff, trg):
        if len(diff) > max_length_code:
            return False, \
                   f"Diff has length {len(diff)} > {max_length_code}"
        if len(trg) > max_length_msg:
            return False, \
                   f"Message has length {len(trg)} > {max_length_msg}"
        return True, None

    return filter_predicate


class CMGDataset(Dataset):
    """Defines a dataset_utils for commit message generation task as torch.utils.raw_data.Dataset"""

    def __init__(self, src_encodings, trg_encodings):
        self.src_encodings = src_encodings
        self.trg_encodings = trg_encodings

    def __getitem__(self, idx):
        src_input_ids = self.src_encodings['input_ids'][idx]
        src_attention_mask = self.src_encodings['attention_mask'][idx]

        trg_input_ids = self.trg_encodings['input_ids'][idx]
        trg_attention_mask = self.trg_encodings['attention_mask'][idx]

        return src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask

    def __len__(self):
        return len(self.trg_encodings['input_ids'])

    @staticmethod
    def load_data(src_tokenizer: RobertaTokenizer, trg_tokenizer: GPT2Tokenizer, path: str, diff_max_len, msg_max_len,
                  verbose=False):
        filter_pred = create_filter_predicate_on_code_and_msg(diff_max_len, msg_max_len)
        prevs = []
        upds = []
        msgs = []
        with open(os.path.join(path, 'diff.txt'), mode='r', encoding='utf-8') as diff, \
                open(os.path.join(path, 'msg.txt'), mode='r', encoding='utf-8') as msg, \
                open(os.path.join(path, 'prev.txt'), mode='r', encoding='utf-8') as prev, \
                open(os.path.join(path, 'updated.txt'), mode='r', encoding='utf-8') as updated:
            for diff_line, msg_line, prev_line, updated_line in zip(diff, msg, prev, updated):
                diff_line, msg_line, prev_line, updated_line = \
                    diff_line.strip(), msg_line.strip(), prev_line.strip(), updated_line.strip()

                is_correct, error = filter_pred(diff_line.split(' '), msg_line.split(' '))

                if not is_correct:
                    if verbose:
                        print(f'Incorrect example is seen. Error: {error}', file=sys.stderr)
                    continue

                prevs.append(prev_line)
                upds.append(updated_line)
                msgs.append(msg_line)
        return CMGDataset(src_encodings=src_tokenizer(prevs, upds, truncation=True,
                                                      padding=True, return_tensors='pt'),
                          trg_encodings=trg_tokenizer(msgs, truncation=True,
                                                      padding=True,
                                                      return_tensors='pt'))


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    train_dataset = CMGDataset.load_data(tokenizer, tokenizer, path=os.path.join('../raw_data/CleanedJiang', 'train'),
                                         diff_max_len=110, msg_max_len=30, verbose=True)
    val_dataset = CMGDataset.load_data(tokenizer, tokenizer, path=os.path.join('../raw_data/CleanedJiang', 'val'),
                                       diff_max_len=110, msg_max_len=30, verbose=True)
    test_dataset = CMGDataset.load_data(tokenizer, tokenizer, path=os.path.join('../raw_data/CleanedJiang', 'test'),
                                        diff_max_len=110, msg_max_len=30, verbose=True)

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))

    print("===Example===")
    src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask = train_dataset[0]
    print("Source input ids")
    print(src_input_ids)
    print("Source attention mask")
    print(src_attention_mask)
    print("Target input ids")
    print(trg_input_ids)
    print("Target attention mask")
    print(trg_attention_mask)
    print()
    print(torch.where(src_input_ids == 0)[0][1])
    src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask = train_dataset[10]
    print(torch.where(src_input_ids == 0)[0][1])
    src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask = train_dataset[1005]
    print(torch.where(src_input_ids == 0)[0][1])
