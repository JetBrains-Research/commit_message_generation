import os
import sys
import json
from typing import List, Dict

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

    def __getitem__(self, idx):
        try:
            msgs = self.ids_to_msgs[self.trg_project_ids[idx]]
        except KeyError:  # there are no train examples for some test repos =(
            msgs = []
        return {"message_input_ids": self.trg_input_ids[idx],
                "history_input_ids": msgs}

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

        trg_input_ids = msg_tokenizer(msgs, truncation=True).input_ids

        # load dict {repo id: all messages from train from this repo (as strings)}
        with open(os.path.join(path, 'train/ids_to_msg.json'), mode='r', encoding='utf-8') as json_file:
            ids_to_msgs = json.load(json_file)

        # encode messages so dict becomes {repo id: all messages from train from this repo (as lists of tokens)}
        for repo_id in ids_to_msgs:
            ids_to_msgs[repo_id] = msg_tokenizer(ids_to_msgs[repo_id], truncation=True).input_ids

        return CMGDatasetWithHistory(trg_input_ids=trg_input_ids,
                                     trg_project_ids=project_ids,
                                     ids_to_msgs=ids_to_msgs)


if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    test_dataset = CMGDatasetWithHistory.load_data(tokenizer, '../raw_data/github_data',
                                                   diff_max_len=110, msg_max_len=30, verbose=True)

    print("Test:", len(test_dataset))

    print("===Example===")
    input = test_dataset[1]

    print("Current message input ids")
    print(input['message_input_ids'])
    print(tokenizer.decode(input['message_input_ids']))

    print("Current history input ids")
    print(input['history_input_ids'])
    print(tokenizer.batch_decode(input['history_input_ids']))
