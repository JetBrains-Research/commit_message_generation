from typing import List
from torch.utils.data import Dataset
from dataset_utils.CommitMessageGenerationDataset import CommitMessageGenerationDataset


def tokenize_git_diff_output_string(git_diff_output: str) -> List[List[str]]:
    tokens = git_diff_output.split()
    tokens_per_line = [[]]
    for token in tokens:
        if token == '<nl>':
            tokens_per_line.append([])
        else:
            tokens_per_line[-1].append(token)
    if len(tokens_per_line[-1]) == 0:
        tokens_per_line = tokens_per_line[:-1]
    return tokens_per_line


def create_filter_predicate_on_length(max_length):
    def filter_predicate(example_data):
        for i, element in enumerate(example_data):
            if len(element) > max_length:
                return False, \
                       f"{i}th element of example has length {len(element)} > {max_length}"
        return True, None
    return filter_predicate


def create_filter_predicate_on_code_and_msg(max_length_code, max_length_msg):
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


def take_part_from_dataset(dataset: CommitMessageGenerationDataset, n: int):
    return CommitMessageGenerationDataset(dataset.src_encodings[:n], dataset.trg_encodings[:n])