import os
import sys

from torch.utils.data import Dataset, Subset
from transformers import RobertaTokenizer


def create_filter_predicate_on_code_and_msg(max_length_code, max_length_msg):
    """Create function to check length of diffs and target messages."""
    def filter_predicate(example_data):
        for i, element in enumerate(example_data):
            if i == 1:
                if len(element) > max_length_msg:
                    return False, \
                           f"{i}th element of example has length {len(element)} > {max_length_msg}"
            else:
                if len(element) > max_length_code:
                    return False, \
                           f"{i}th element of example has length {len(element)} > {max_length_code}"
        return True, None
    return filter_predicate


class CMGDataset(Dataset):
    """Defines a dataset_utils for commit message generation task as torch.utils.raw_data.Dataset"""
    def __init__(self, src_encodings, trg_encodings):
        self.src_encodings = src_encodings
        self.trg_encodings = trg_encodings

    def __getitem__(self, idx):
        src = {key: val[idx].clone().detach() for key, val in self.src_encodings.items()}
        trg = {key: val[idx].clone().detach() for key, val in self.trg_encodings.items()}
        return src, trg

    def __len__(self):
        return len(self.trg_encodings['input_ids'])

    @staticmethod
    def load_data(codebert_tokenizer: RobertaTokenizer, path: str, diff_max_len, msg_max_len, verbose=False):
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

                is_correct, error = filter_pred((diff_line.split(' '), msg_line.split(' ')))

                if not is_correct:
                    if verbose:
                        print(f'Incorrect example is seen. Error: {error}', file=sys.stderr)
                    continue

                prevs.append(prev_line)
                upds.append(updated_line)
                msgs.append(msg_line)

        return CMGDataset(src_encodings=codebert_tokenizer(prevs, upds, truncation=True,
                                                           padding=True, return_tensors='pt'),
                          trg_encodings=codebert_tokenizer(msgs, truncation=True,
                                                           padding=True,
                                                           return_tensors='pt'))

    @staticmethod
    def take_first_n_from_dataset(dataset, n: int):
        return Subset(dataset, range(n))


if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    train_dataset = CMGDataset.load_data(tokenizer, path=os.path.join('../raw_data/CleanedJiang', 'train'),
                                         diff_max_len=100, msg_max_len=30, verbose=True)
    val_dataset = CMGDataset.load_data(tokenizer, path=os.path.join('../raw_data/CleanedJiang', 'val'),
                                       diff_max_len=100, msg_max_len=30, verbose=True)
    test_dataset = CMGDataset.load_data(tokenizer, path=os.path.join('../raw_data/CleanedJiang', 'test'),
                                        diff_max_len=100, msg_max_len=30, verbose=True)

    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))

    print("Example")
    print(train_dataset[0])
