import os
import sys

import torch
from torch.utils.data import Dataset

from dataset_utils.utils import create_filter_predicate_on_code_and_msg
from Config import Config

from transformers import RobertaTokenizer


class CommitMessageGenerationDataset(Dataset):
    """Defines a dataset for commit message generation task as torch.utils.data.Dataset"""

    def __init__(self, src_encodings, trg_encodings):
        self.src_encodings = src_encodings
        self.trg_encodings = trg_encodings

    def __getitem__(self, idx):
        # TODO: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach()
        #  or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor)
        src = {key: torch.tensor(val[idx]) for key, val in self.src_encodings.items()}
        src['target'] = {key: torch.tensor(val[idx]) for key, val in self.trg_encodings.items()}
        return src

    def __len__(self):
        return len(self.src_encodings.input_ids)

    @staticmethod
    def load_data(path: str, config: Config, small=False, verbose=False):
        codebert_tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

        filter_pred = create_filter_predicate_on_code_and_msg(config['TOKENS_CODE_CHUNK_MAX_LEN'],
                                                              config['MSG_MAX_LEN'])
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
        if small:
            return CommitMessageGenerationDataset(
                src_encodings=codebert_tokenizer(prevs[:50], upds[:50], truncation=True,
                                                 padding=True, return_tensors='pt'),
                trg_encodings=codebert_tokenizer(msgs[:50], truncation=True, padding=True,
                                                 return_tensors='pt'))
        return CommitMessageGenerationDataset(src_encodings=codebert_tokenizer(prevs, upds, truncation=True,
                                                                               padding=True, return_tensors='pt'),
                                              trg_encodings=codebert_tokenizer(msgs, truncation=True,
                                                                               padding=True,
                                                                               return_tensors='pt'))


if __name__ == "__main__":
    config = Config()

    train_dataset = CommitMessageGenerationDataset.load_data(path=os.path.join(config['DATASET_ROOT'], 'train'),
                                                             config=config)
    val_dataset = CommitMessageGenerationDataset.load_data(path=os.path.join(config['DATASET_ROOT'], 'val'),
                                                           config=config)
    test_dataset = CommitMessageGenerationDataset.load_data(path=os.path.join(config['DATASET_ROOT'], 'test'),
                                                            config=config)
    print("Train:", len(train_dataset))
    print("Val:", len(val_dataset))
    print("Test:", len(test_dataset))

    print("Example")
    print(train_dataset[0])
