import os
import sys
import json
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


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


class CMGDatasetWithHistory(Dataset):
    """Defines a dataset_utils for commit message generation task as torch.utils.raw_data.Dataset.
    This version doesn't use diffs but concatenates each message to history from the same repo."""

    def __init__(self, trg_input_ids: List[List[int]], trg_project_ids: List[str],
                 ids_to_msgs: Dict[str, List[List[int]]]):
        self.trg_input_ids = trg_input_ids
        self.trg_project_ids = trg_project_ids
        self.ids_to_msgs = ids_to_msgs
        # all tensors in dataset will have this sequence length
        self.max_len = 1024  # max gpt-2 input length - not optimal in terms of memory, choose some number myself?

    def construct_tensor_from_example_and_history(self,
                                                  trg_input_ids: List[int],
                                                  trg_history: List[List[int]],
                                                  pad_token_id: int = 50256) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Helper function.
        Concatenate trg_input_ids to trg_history, pad all examples to max_length and convert to torch.Tensor
        """

        # concatenate history examples with current input ids (checking that resulting length is <= max_len)
        ids = [trg_input_ids]
        labels = [trg_input_ids]
        cur_len = len(trg_input_ids)
        for history_input_ids in trg_history[::-1]:
            if cur_len + len(history_input_ids) > self.max_len:
                break
            cur_len += len(history_input_ids)
            ids.insert(0, history_input_ids)
            labels.insert(0, [-100 for _ in history_input_ids])

        # flatten everything into one sequence of ids and convert to tensor of torch.int
        ids = torch.tensor([ex for sublist in ids for ex in sublist], dtype=torch.int64)
        labels = torch.tensor([ex for sublist in labels for ex in sublist], dtype=torch.int64)
        # create ones for attention mask
        mask = torch.ones_like(ids)

        # pad ids with pad_token_id (which doesn't really matter) and mask with zeros
        ids = torch.nn.functional.pad(ids, pad=(0, self.max_len - ids.numel()), mode='constant', value=pad_token_id)
        labels = torch.nn.functional.pad(labels, pad=(0, self.max_len - labels.numel()), mode='constant', value=-100)
        mask = torch.nn.functional.pad(mask, pad=(0, self.max_len - mask.numel()), mode='constant', value=0)

        return ids, mask, labels

    def __getitem__(self, idx):
        try:
            msgs = self.ids_to_msgs[self.trg_project_ids[idx]]
        except KeyError:  # there are no train examples for some test repos =(
            msgs = []
        return self.construct_tensor_from_example_and_history(self.trg_input_ids[idx],
                                                              msgs)

    def __len__(self):
        return len(self.trg_input_ids)

    @staticmethod
    def load_data(msg_tokenizer: GPT2Tokenizer, path: str, diff_max_len, msg_max_len,
                  verbose=False):
        filter_pred = create_filter_predicate_on_code_and_msg(diff_max_len, msg_max_len)

        project_ids = []
        msgs = []

        with open(os.path.join(path, 'test/diff.txt'), mode='r', encoding='utf-8') as diff, \
                open(os.path.join(path, 'test/msg.txt'), mode='r', encoding='utf-8') as msg, \
                open(os.path.join(path, 'test/projectIds.txt'), mode='r', encoding='utf-8') as proj_id:

            for diff_line, msg_line, proj_id_line in zip(diff, msg, proj_id):

                diff_line, msg_line, proj_id_line = diff_line.strip(), msg_line.strip(), proj_id_line.strip()

                # drop examples with number of tokens > max_len
                is_correct, error = filter_pred(diff_line.split(' '), msg_line.split(' '))
                if not is_correct:
                    if verbose:
                        print(f'Incorrect example is seen. Error: {error}', file=sys.stderr)
                    continue

                project_ids.append(proj_id_line)
                msgs.append(msg_line)

        trg_input_ids = msg_tokenizer(msgs, add_special_tokens=True).input_ids

        # load dict {repo id: all messages from train from this repo (as strings)}
        with open(os.path.join(path, 'train/ids_to_msg.json'), mode='r', encoding='utf-8') as json_file:
            ids_to_msgs = json.load(json_file)

        # encode messages so dict becomes {repo id: all messages from train from this repo (as lists of tokens)}
        for repo_id in ids_to_msgs:
            ids_to_msgs[repo_id] = msg_tokenizer(ids_to_msgs[repo_id], add_special_tokens=True).input_ids

        return CMGDatasetWithHistory(trg_input_ids=trg_input_ids,
                                     trg_project_ids=project_ids,
                                     ids_to_msgs=ids_to_msgs)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    test_dataset = CMGDatasetWithHistory.load_data(tokenizer, '../raw_data/CleanedJiang',
                                                   diff_max_len=110, msg_max_len=30, verbose=True)

    print("Test:", len(test_dataset))
    print("===Example===")
    trg_input_ids, trg_attention_mask, trg_labels = test_dataset[0]
    print("Target input ids")
    print(trg_input_ids)
    print("Target attention mask")
    print(trg_attention_mask)
    print("Labels")
    print(trg_labels)
