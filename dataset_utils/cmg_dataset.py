import os
import sys

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
    def __init__(self, diff_input_ids, msg_input_ids):
        self.diff_input_ids = diff_input_ids
        self.msg_input_ids = msg_input_ids

    def __getitem__(self, idx):
        return {"diff_input_ids": self.diff_input_ids[idx], "message_input_ids": self.msg_input_ids[idx]}

    def __len__(self):
        return len(self.msg_input_ids)

    @staticmethod
    def load_data(tokenizer: RobertaTokenizer, path: str, diff_max_len, msg_max_len,
                  verbose=False):
        filter_pred = create_filter_predicate_on_code_and_msg(diff_max_len, msg_max_len)
        new_diffs = []
        msgs = []

        with open(os.path.join(path, 'diff.txt'), mode='r', encoding='utf-8') as diff, \
                open(os.path.join(path, 'msg.txt'), mode='r', encoding='utf-8') as msg, \
                open(os.path.join(path, 'new_diff.txt'), mode='r', encoding='utf-8') as new_diff:
            for diff_line, msg_line, new_diff_line in zip(diff, msg, new_diff):
                diff_line, msg_line, new_diff_line = diff_line.strip(), msg_line.strip(), new_diff_line.strip()

                is_correct, error = filter_pred(diff_line.split(' '), msg_line.split(' '))

                if not is_correct:
                    if verbose:
                        print(f'Incorrect example is seen. Error: {error}', file=sys.stderr)
                    continue

                new_diffs.append(new_diff_line)
                msgs.append(msg_line)

        new_diff_enc = tokenizer(new_diffs, add_special_tokens=True)

        msg_enc = tokenizer(msgs, add_special_tokens=True)

        return CMGDataset(diff_input_ids=new_diff_enc.input_ids,
                          msg_input_ids=msg_enc.input_ids)


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    train_dataset = CMGDataset.load_data(tokenizer, path=os.path.join('../raw_data/CleanedJiang', 'train'),
                                         diff_max_len=110, msg_max_len=30, verbose=True)
    val_dataset = CMGDataset.load_data(tokenizer, path=os.path.join('../raw_data/CleanedJiang', 'val'),
                                       diff_max_len=110, msg_max_len=30, verbose=True)
    test_dataset = CMGDataset.load_data(tokenizer, path=os.path.join('../raw_data/CleanedJiang', 'test'),
                                        diff_max_len=110, msg_max_len=30, verbose=True)

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))

    print("===Example===")
    src_input_ids, trg_input_ids = train_dataset[0]
    print("Source input ids")
    print(src_input_ids)
    print("Target input ids")
    print(trg_input_ids)
    print()