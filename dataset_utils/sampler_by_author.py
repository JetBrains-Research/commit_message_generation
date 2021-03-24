from torch.utils.data import Sampler
import torch
from typing import Iterator, List, Sized
from itertools import chain


class RandomSamplerByAuthor(Sampler):
    r"""Samples elements (kind of) randomly.
    To keep history consistent, the order within each author's commits is preserved.
    At each step random author is chosen and id of next commit of this author is returned.

        Args:
            data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source: Sized,
                 generator=None):
        self.data_source = data_source
        self.generator = generator

    def __iter__(self):
        if self.generator is None:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        else:
            generator = self.generator

        x = self.data_source.get_iterators_by_authors()
        while len(x) > 0:
            idx = torch.randint(high=len(x), size=(), generator=generator).data
            try:
                yield next(x[idx])
            except StopIteration:
                del x[idx]

    def __len__(self):
        return len(self.data_source)


class SamplerByAuthor(Sampler):
    r"""Samples elements in the following order:
    - all commits of the first author
    - then all commits of the seconda author
    - and so on...
    useful at validation & test time to avoid randomness

    TODO: maybe it's better to sample 1 element from each author always in the same order?

    ugly code snippet
    idx = 0
        while len(data_iters) > 0:
            try:
                yield next(data_iters[idx])
                idx += 1
            except StopIteration:
                del data_iters[idx]
            except IndexError:
                idx = 0

        Args:
            data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source: Sized):
        self.data_source = data_source

    def __iter__(self):
        return iter(chain.from_iterable(self.data_source.get_iterators_by_authors()))

    def __len__(self):
        return len(self.data_source)


if __name__ == "__main__":
    from transformers import RobertaTokenizer, GPT2Tokenizer
    from dataset_utils.cmg_dataset_w_history import CMGDatasetWithHistory
    from dataset_utils.data_collator_w_history import DataCollatorWithHistory
    from torch.utils.data import DataLoader

    diff_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    msg_tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    msg_tokenizer.pad_token = msg_tokenizer.unk_token

    test_dataset = CMGDatasetWithHistory.load_data(diff_tokenizer, msg_tokenizer,
                                                   path='../raw_data/CleanedJiang/test.csv')

    data_collator = DataCollatorWithHistory(tokenizer=msg_tokenizer, max_len=1024)
    sampler = SamplerByAuthor(test_dataset)

    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=data_collator, sampler=sampler)
    i = 0
    for batch in test_dataloader:
        print("Message (history + cur_msg)")
        print(msg_tokenizer.batch_decode(batch['msg_input_ids'], skip_special_tokens=True))
        print("Generation (history)")
        print(msg_tokenizer.batch_decode(batch['generation_input_ids'], skip_special_tokens=True))
        print()
        i += 1

