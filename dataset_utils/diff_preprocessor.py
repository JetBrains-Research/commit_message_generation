import itertools
import os
from typing import Tuple, List


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


class DiffPreprocessor:
    """Class to process git diff into prev and updated."""
    @staticmethod
    def get_prev_and_updated(git_diff_output: str, verbose=False) -> Tuple[str, str]:
        """
        Generates prev and updated code from output of git diff command.
        :param git_diff_output: output of git diff command
        :return: [prev, updated]
        """
        tokens_per_line = tokenize_git_diff_output_string(git_diff_output)

        prev_lines, updated_lines = [], []
        was_special_keyword_modification = False
        for tokens_in_line in tokens_per_line:
            if tokens_in_line[0] == 'mmm':
                prev_lines.append(tokens_in_line[1:])
                was_special_keyword_modification = True
            elif tokens_in_line[0] == 'ppp':
                updated_lines.append(tokens_in_line[1:])
                was_special_keyword_modification = True
            elif tokens_in_line[:3] == ['new', 'file', 'mode']:
                updated_lines.append(['new', 'file'])
                was_special_keyword_modification = True
            elif tokens_in_line[:3] == ['deleted', 'file', 'mode']:
                updated_lines.append(['deleted', 'file'])
                was_special_keyword_modification = True
            elif tokens_in_line[:2] == ['rename', 'from']:
                prev_lines.append(tokens_in_line)
                was_special_keyword_modification = True
            elif tokens_in_line[:2] == ['rename', 'to']:
                updated_lines.append(tokens_in_line)
                was_special_keyword_modification = True
            elif tokens_in_line[:2] == ['old', 'mode']:
                prev_lines.append(['old', 'mode'])
                was_special_keyword_modification = True
            elif tokens_in_line[:2] == ['new', 'mode']:
                updated_lines.append(['new', 'mode'])
                was_special_keyword_modification = True
            elif tokens_in_line[0] == '-':
                prev_lines.append(tokens_in_line[1:])
            elif tokens_in_line[0] == '+':
                updated_lines.append(tokens_in_line[1:])
            elif tokens_in_line[0] == 'index' or tokens_in_line[:2] == ['similarity', 'index']:
                continue
            else:
                prev_lines.append(tokens_in_line)
                updated_lines.append(tokens_in_line)

        prev = ' '.join(itertools.chain(*[line + ['\\n'] for line in prev_lines]))
        updated = ' '.join(itertools.chain(*[line + ['\\n'] for line in updated_lines]))
        if not was_special_keyword_modification:
            print(f'No special keyword found for diff: {git_diff_output}')
        if prev == updated:
            print(f'Prev and updated are the same for diff: {git_diff_output}')
        return prev + '\n', updated + '\n'

    @staticmethod
    def get_prev_and_updated_for_diffs(git_diff_outputs: List[str]) -> List[Tuple[str, str]]:
        prev_res = []
        upd_res = []
        for git_diff_output in git_diff_outputs:
            prev, updated = DiffPreprocessor.get_prev_and_updated(git_diff_output)
            if prev is not None and updated is not None:
                prev_res.append(prev)
                upd_res.append(updated)
        return prev_res, upd_res

    @staticmethod
    def create_files(ds_root_path: str):
        for part in ['train', 'val', 'test']:
            cur_path = os.path.join(ds_root_path, part)
            with open(os.path.join(cur_path, 'diff.txt')) as diff_file, \
                    open(os.path.join(cur_path, 'prev.txt'), 'w') as prev_file, \
                    open(os.path.join(cur_path, 'updated.txt'), 'w') as upd_file:
                prev, updated = DiffPreprocessor.get_prev_and_updated_for_diffs(diff_file.readlines())
                prev_file.writelines(prev)
                upd_file.writelines(updated)


if __name__ == "__main__":
    dataset_path = '../raw_data/CleanedJiang/'
    DiffPreprocessor.create_files(dataset_path)
