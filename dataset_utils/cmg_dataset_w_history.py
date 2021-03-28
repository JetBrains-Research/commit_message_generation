import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, RobertaTokenizer, GPT2Tokenizer


class CMGDatasetWithHistory(Dataset):
    """Defines a map-style dataset for commit message generation task.
    This version provides history from the same author for each commit.
    Therefore for each commit it's author and it's position inside it's author's history are keeped.
    """

    def __init__(self,
                 diff_input_ids: torch.Tensor,
                 diff_attention_mask: torch.Tensor,
                 msg_input_ids: List[List[int]],
                 msg_authors: List[Any],
                 msg_positions_in_history: List[int],
                 history: Dict[Any, List[List[int]]],
                 iters: List[List[int]]):

        self.diff_input_ids = diff_input_ids
        self.diff_attention_mask = diff_attention_mask

        self.msg_input_ids = msg_input_ids
        self.msg_authors = msg_authors

        self.msg_positions_in_history = msg_positions_in_history
        self.history = history

        self._iters = iters

    def __getitem__(self, idx):
        try:
            history_idx = self.msg_positions_in_history[idx]
            history_msgs = self.history[self.msg_authors[idx]][:history_idx]
        except KeyError:  # if there are no examples for this author =(
            history_msgs = []
        return {"diff_input_ids": self.diff_input_ids[idx],
                "diff_attention_mask": self.diff_attention_mask[idx],
                "msg_input_ids": self.msg_input_ids[idx],
                "history_input_ids": history_msgs}

    def __len__(self):
        return len(self.msg_input_ids)

    def get_iterators_by_authors(self):
        return [iter(value) for value in self._iters]

    @staticmethod
    def load_data(diff_tokenizer: PreTrainedTokenizerBase,
                  msg_tokenizer: PreTrainedTokenizerBase,
                  path: str):

        df = pd.read_csv(path, names=['diff', 'msg', 'id'])
        df['id_position'] = df.groupby('id').cumcount()

        diffs = df['diff'].tolist()
        msgs = df['msg'].tolist()
        ids = df['id'].tolist()
        positions_in_history = df['id_position'].tolist()
        iters = [value for _, value in df.groupby('id').groups.items()]

        diff_enc = diff_tokenizer(diffs, truncation=True, padding=True,
                                  return_tensors='pt', add_special_tokens=True, max_length=500)
        msg_input_ids = msg_tokenizer(msgs, truncation=True).input_ids

        # create history as dict {repo id: all messages from this repo (as lists of tokens)}
        history = defaultdict(list)
        for msg, id in zip(msg_input_ids, ids):
            history[id].append(msg)

        return CMGDatasetWithHistory(diff_input_ids=diff_enc.input_ids,
                                     diff_attention_mask=diff_enc.attention_mask,
                                     msg_input_ids=msg_input_ids,
                                     msg_authors=ids,
                                     msg_positions_in_history=positions_in_history,
                                     history=history,
                                     iters=iters)


if __name__ == "__main__":
    diff_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    msg_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")

    train_dataset = CMGDatasetWithHistory.load_data(diff_tokenizer, msg_tokenizer,
                                                    '../raw_data/CleanedJiang/train.csv')

    test_dataset = CMGDatasetWithHistory.load_data(diff_tokenizer, msg_tokenizer,
                                                   '../raw_data/CleanedJiang/test.csv')

    print("Train:", len(train_dataset))
    print("Test:", len(test_dataset))
    print()

    for i in range(10):
        print(f"===Example {i+1}===")
        print()

        idx = np.random.randint(len(test_dataset))
        input = test_dataset[idx]

        print("Current diff input ids")
        print(diff_tokenizer.decode(input['diff_input_ids']))
        print()
        print("Current message input ids")
        print(input['msg_input_ids'])
        print(msg_tokenizer.decode(input['msg_input_ids']))
        print()
        print("Current history input ids")
        print(input['history_input_ids'])
        print(msg_tokenizer.batch_decode(input['history_input_ids']))
        print()
        print("Current position in history")
        print(test_dataset.msg_positions_in_history[idx])
        print()
        print("Current author")
        print(test_dataset.msg_authors[idx])
        print()
        print("Full history for current author")
        print(test_dataset.history[test_dataset.msg_authors[idx]])
        print(msg_tokenizer.batch_decode(test_dataset.history[test_dataset.msg_authors[idx]]))
        print()
