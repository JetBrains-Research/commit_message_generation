import os
import sys

from torch.utils.data import Dataset
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

    def __init__(self, diff_input_ids, diff_attention_mask, msg_input_ids_unprocessed):
        self.diff_input_ids = diff_input_ids
        self.diff_attention_mask = diff_attention_mask
        self.msg_input_ids_unprocessed = msg_input_ids_unprocessed

    def __getitem__(self, idx):
        return {"diff_input_ids": self.diff_input_ids[idx],  # tensor
                "diff_attention_mask": self.diff_attention_mask[idx],  # tensor
                "msg_input_ids_unprocessed": self.msg_input_ids_unprocessed[idx]}  # list of lists

    def __len__(self):
        return len(self.diff_input_ids)

    @staticmethod
    def load_data(src_tokenizer: RobertaTokenizer, trg_tokenizer: GPT2Tokenizer, path: str, diff_max_len, msg_max_len,
                  verbose=False):
        filter_pred = create_filter_predicate_on_code_and_msg(diff_max_len, msg_max_len)

        diffs = []
        msgs = []

        with open(os.path.join(path, 'diff.txt'), mode='r', encoding='utf-8') as diff, \
                open(os.path.join(path, 'msg.txt'), mode='r', encoding='utf-8') as msg:
            for diff_line, msg_line in zip(diff, msg):
                diff_line, msg_line = diff_line.strip(), msg_line.strip()

                is_correct, error = filter_pred(diff_line.split(' '), msg_line.split(' '))

                if not is_correct:
                    if verbose:
                        print(f'Incorrect example is seen. Error: {error}', file=sys.stderr)
                    continue

                diffs.append(diff_line)
                msgs.append(msg_line)

        diff_enc = src_tokenizer(diffs, truncation=True, padding=True,
                                 return_tensors='pt', add_special_tokens=True)
        msg_enc_unprocessed = trg_tokenizer(msgs)

        return CMGDataset(diff_input_ids=diff_enc.input_ids,
                          diff_attention_mask=diff_enc.attention_mask,
                          msg_input_ids_unprocessed=msg_enc_unprocessed.input_ids)


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    train_dataset = CMGDataset.load_data(tokenizer, tokenizer, path=os.path.join('../raw_data/github_data', 'train'),
                                         diff_max_len=110, msg_max_len=30, verbose=True)
    test_dataset = CMGDataset.load_data(tokenizer, tokenizer, path=os.path.join('../raw_data/github_data', 'test'),
                                        diff_max_len=110, msg_max_len=30, verbose=True)

    print("Train:", len(train_dataset))
    print("Test:", len(test_dataset))
