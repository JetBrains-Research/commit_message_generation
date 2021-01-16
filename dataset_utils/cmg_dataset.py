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

    def __init__(self, src_input_ids, src_attention_mask, src_token_type_ids, trg_input_ids, trg_attention_mask):
        self.src_input_ids = src_input_ids
        self.src_attention_mask = src_attention_mask
        self.src_token_type_ids = src_token_type_ids
        self.trg_input_ids = trg_input_ids
        self.trg_attention_mask = trg_attention_mask

    def __getitem__(self, idx):
        return self.src_input_ids[idx], self.src_attention_mask[idx], self.src_token_type_ids[idx],\
               self.trg_input_ids[idx], self.trg_attention_mask[idx]

    def __len__(self):
        return len(self.trg_input_ids)

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

        enc = src_tokenizer(prevs, upds, truncation=True, padding=True, return_tensors='pt', add_special_tokens=True)
        # get input ids and attention mask from encoding
        src_input_ids = enc.input_ids
        src_attention_mask = enc.attention_mask

        # construct token_type_ids
        src_token_type_ids = torch.ones_like(enc.input_ids)
        # find where first sentence ends (each sequence contains three eos tokens, we are interested in the 2nd)
        end = torch.where(enc.input_ids == src_tokenizer.eos_token_id)[1][1::3]
        for i, ind in enumerate(end):
            src_token_type_ids[i, :ind + 1] = torch.zeros((1, ind + 1))
        msg_enc = trg_tokenizer(msgs, truncation=True, padding=True, return_tensors='pt')
        return CMGDataset(src_input_ids=src_input_ids,
                          src_attention_mask=src_attention_mask,
                          src_token_type_ids=src_token_type_ids,
                          trg_input_ids=msg_enc.input_ids,
                          trg_attention_mask=msg_enc.attention_mask)


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
    src_input_ids, src_attention_mask, src_token_type_ids, trg_input_ids, trg_attention_mask = train_dataset[0]
    print("Source input ids")
    print(src_input_ids)
    print("Source attention mask")
    print(src_attention_mask)
    print("Source token type ids")
    print(src_token_type_ids)
    print("Target input ids")
    print(trg_input_ids)
    print("Target attention mask")
    print(trg_attention_mask)
    print()