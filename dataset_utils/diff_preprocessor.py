import itertools
import os
from typing import Tuple, List
from dataset_utils.edit_distance_utils import align_lists


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

        prev_lines, updated_lines, removed, added = [], [], [], []
        was_special_keyword_modification = False
        rename_len = None
        another_file_name_len = None

        for tokens_in_line in tokens_per_line:
            if tokens_in_line[0] == 'mmm':
                # name of changed file
                # example: mmm a / telecomm / java / android / telecomm / Connection . java
                # example (if new file was created): mmm / dev / null
                if tokens_in_line[1] == 'a':
                    prev_lines.append(tokens_in_line[3:])
                    another_file_name_len = len(tokens_in_line) - 3
                else:
                    prev_lines.append(tokens_in_line[2:])
                was_special_keyword_modification = True

            elif tokens_in_line[0] == 'ppp':
                # name of changed file
                # example: ppp b / telecomm / java / android / telecomm / Connection . java
                # example (if file was deleted):  ppp / dev / null
                if tokens_in_line[1] == 'b':
                    updated_lines.append(tokens_in_line[3:])
                    if prev_lines[-1] == ['dev', '/', 'null']:
                        prev_lines[-1].extend(["<empty>" for _ in range(len(tokens_in_line) - 3)])
                else:
                    updated_lines.append(tokens_in_line[2:])
                    cur_len = len(tokens_in_line) - 2
                    updated_lines[-1].extend(["<empty>" for _ in range(another_file_name_len - cur_len)])
                was_special_keyword_modification = True

            elif tokens_in_line[:3] == ['new', 'file', 'mode']:
                # line in git diff when new file is created
                # example: new file mode 100644
                prev_lines.append(["<empty>", "<empty>"])
                updated_lines.append(['new', 'file'])
                was_special_keyword_modification = True

            elif tokens_in_line[:3] == ['deleted', 'file', 'mode']:
                # line in git diff when file is deleted
                # example: deleted file mode 100644
                prev_lines.append(["<empty>", "<empty>"])
                updated_lines.append(['deleted', 'file'])
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['rename', 'from']:
                # line in git diff when file was renamed (old name)
                # example: rename from src / forge / resources / worldedit . properties
                prev_lines.append(tokens_in_line)
                if rename_len:
                    cur_len = len(tokens_in_line) - 2
                    if cur_len < rename_len:
                        prev_lines[-1].extend(["<empty>" for _ in range(rename_len - cur_len)])
                    else:
                        updated_lines[-1].extend(["<empty>" for _ in range(cur_len - rename_len)])
                    rename_len = None
                else:
                    rename_len = len(tokens_in_line) - 2
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['rename', 'to']:
                # line in git diff when file was renamed (new name)
                # example: rename to src / forge / resources / defaults / worldedit . properties
                updated_lines.append(tokens_in_line)
                if rename_len:
                    cur_len = len(tokens_in_line) - 2
                    if cur_len < rename_len:
                        updated_lines[-1].extend(["<empty>" for _ in range(rename_len - cur_len)])
                    else:
                        prev_lines[-1].extend(["<empty>" for _ in range(cur_len - rename_len)])
                    rename_len = None
                else:
                    rename_len = len(tokens_in_line) - 2
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['old', 'mode']:
                # line in git diff when file mode was changed
                # example: old mode 100644
                # 644=rw-r--r--
                prev_lines.append(tokens_in_line)
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['new', 'mode']:
                # line in git diff when file mode was changed
                # example: new mode 100755
                # 755=rwxr-xr-x
                updated_lines.append(tokens_in_line)
                was_special_keyword_modification = True

            elif tokens_in_line[0] == '-':
                # lines that were removed
                # example: - version = ' 2 . 0 . 2 '
                removed.append(tokens_in_line)

            elif tokens_in_line[0] == '+':
                # lines that were added
                # example: + version = ' 2 . 0 . 3 '
                added.append(tokens_in_line)

            elif tokens_in_line[0] == 'index' or tokens_in_line[:2] == ['similarity', 'index']:
                # some special info that we are not interested in
                # example 1: index 0000000 . . 3f26e45
                # example 2: similarity index 100 %
                continue

            else:
                # all other cases
                # case 1: line that was not changed (drop them)
                # case 2: Binary files a / dependencies / windows / sumatra / SumatraPDF . exe and / dev / null differ
                if tokens_in_line[:2] == ["Binary", "files"]:
                    prev_lines.append(tokens_in_line)
                    updated_lines.append(tokens_in_line)

        # align removed and added
        removed, added = align_lists(removed, added)
        for prev_line, upd_line in zip(removed, added):
            prev_lines.append(prev_line)
            updated_lines.append(upd_line)

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