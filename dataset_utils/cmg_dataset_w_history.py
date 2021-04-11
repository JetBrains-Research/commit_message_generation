import os
import json
from typing import List, Dict
from torch.utils.data import IterableDataset
from transformers import RobertaTokenizer, GPT2Tokenizer


class CMGDatasetWithHistory(IterableDataset):
    """Defines a map-style dataset for commit message generation task.
    This version provides history from the same author for each commit.
    Therefore for each commit it's author and it's position inside it's author's history are keeped.
    """

    def __init__(self,
                 filename: str,
                 history: Dict[str, List[int]]):

        self._filename = filename
        with open(filename, 'r') as f:
            self._len = sum(1 for _ in f)
        self.history = history
        super(CMGDatasetWithHistory).__init__()

    def __iter__(self):
        with open(self._filename) as f:
            for line in f:
                line = json.loads(line.strip())
                yield {'diff_input_ids': line['diff_input_ids'],
                       'msg_input_ids': self.history[str(line['author'])][line['pos_in_history']],
                       'history_input_ids': self.history[str(line['author'])][:line['pos_in_history']]}

    @staticmethod
    def load_data(dataset_root: str,
                  part: str):

        with open(os.path.join(dataset_root, f'{part}_history.json'), 'r') as infile:
            history = json.load(infile)

        return CMGDatasetWithHistory(os.path.join(dataset_root, f'{part}.json'), history)


if __name__ == "__main__":
    diff_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    msg_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    test_dataset = CMGDatasetWithHistory.load_data('../raw_data/github_data/',
                                                   'test')

    print("Test")
    print()

    i = 0
    for input in test_dataset:
        print(f'===== Example {i+1} =====')
        print("Current diff input ids")
        print(diff_tokenizer.decode(input['diff_input_ids'], skip_special_tokens=True).replace('\n', '\\n'))
        print()
        print("Current message input ids")
        print(msg_tokenizer.decode(input['msg_input_ids'], skip_special_tokens=True).replace('\n', '\\n'))
        print()
        print("Current history input ids")
        print(msg_tokenizer.batch_decode(input['history_input_ids'], skip_special_tokens=True))
        print()

        i += 1
        if i == 10:
            break
