import itertools
from collections import defaultdict
import os
from typing import List, Dict
import pandas as pd
import json
from tqdm.notebook import tqdm
from transformers import PreTrainedTokenizerBase, RobertaTokenizerFast, GPT2TokenizerFast


class DataPreprocessor:
    """
    Class to process dataset files.
    Current version does the following:
    1) Replaces <nl> with \n in diffs and messages
    2) Removes unchanged lines from diffs
    3) Tokenizes diffs and messages
    """

    @staticmethod
    def _tokenize_git_diff_output_string(diff: str) -> List[List[str]]:
        lines = [line.split() for line in diff.split('<nl>')]
        return lines

    @staticmethod
    def preprocess_diff(git_diff_output: str) -> str:
        """
        1) Removes some special tokens
        2) Removes non-changed lines from diffs
        3) Replaces <nL> with \n
        """
        tokens_per_line = DataPreprocessor._tokenize_git_diff_output_string(git_diff_output)

        diff_lines = []
        was_special_keyword_modification = False

        for tokens_in_line in tokens_per_line:
            if len(tokens_in_line) == 0:
                # remove empty lines
                continue

            elif tokens_in_line[0] == '<FILE>':
                # name of changed file
                # example: <FILE> telecomm / java / android / telecomm / Connection . java
                diff_lines.append(tokens_in_line[1:])
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['new', 'file']:
                # line in git diff when new file is created
                # example: new file
                diff_lines.append(['new', 'file'])
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['deleted', 'file']:
                # line in git diff when file is deleted
                # example: deleted file
                diff_lines.append(['deleted', 'file'])
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['rename', 'from']:
                # line in git diff when file was renamed (old name)
                # example: rename from src / forge / resources / worldedit . properties
                diff_lines.append(tokens_in_line)
                was_special_keyword_modification = True

            elif tokens_in_line[:2] == ['rename', 'to']:
                # line in git diff when file was renamed (new name)
                # example: rename to src / forge / resources / defaults / worldedit . properties
                diff_lines.append(tokens_in_line)
                was_special_keyword_modification = True

            elif tokens_in_line[0] == '-':
                # lines that were removed
                # example: - version = ' 2 . 0 . 2 '
                diff_lines.append(tokens_in_line)

            elif tokens_in_line[0] == '+':
                # lines that were added
                # example: + version = ' 2 . 0 . 3 '
                diff_lines.append(tokens_in_line)

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
                    diff_lines.append(tokens_in_line)

        diff = ' '.join(itertools.chain(*[line + ['\n'] for line in diff_lines]))

        return diff + '\n'

    @staticmethod
    def preprocess_diffs(git_diff_outputs: List[str]) -> List[str]:
        diff_res = []
        for git_diff_output in git_diff_outputs:
            diff = DataPreprocessor.preprocess_diff(git_diff_output)
            if diff is not None:
                diff_res.append(diff)
        return diff_res

    @staticmethod
    def preprocess_messages(msgs: List[str]) -> List[str]:
        resulting_msgs = []
        for msg in msgs:
            resulting_msgs.append(msg.replace('<nl>', '\n'))
        return resulting_msgs

    @staticmethod
    def tokenize_diffs(diffs: List[str], tokenizer: PreTrainedTokenizerBase) -> List[List[int]]:
        res = []
        for diff in tqdm(diffs):
          cur_res = tokenizer(diff, padding=True, truncation=True, add_special_tokens=True, max_length=500, return_attention_mask=False)
          res.append(cur_res.input_ids)
        return res

    @staticmethod
    def tokenize_messages(msgs: List[str], tokenizer: PreTrainedTokenizerBase) -> List[List[int]]:
      res = []
      for msg in tqdm(msgs):
          cur_res = tokenizer(msg, truncation=True, return_attention_mask=False).input_ids
          res.append(cur_res)
      return res

    @staticmethod
    def create_files(ds_root_path: str):
        for part in ['train', 'val', 'test']:
            print(f'Processing {part}')

            print('Reading data')
            if f'processed_{part}.csv' not in os.listdir(os.path.join(ds_root_path)):
              df = pd.read_csv(os.path.join(ds_root_path, f'{part}.csv'))

              print('Processing data')
              df['diff'] = DataPreprocessor.preprocess_diffs(df['diff'].tolist())
              df['message'] = DataPreprocessor.preprocess_messages(df['message'].tolist())
              df['pos_in_history'] = df.groupby('author').cumcount()
              df.to_csv(os.path.join(ds_root_path, f'processed_{part}.csv'), index=None)
            else:
              df = pd.read_csv(os.path.join(ds_root_path, f'processed_{part}.csv'))

            print('Tokenizing diffs')
            diff_input_ids = DataPreprocessor.tokenize_diffs(df['diff'].tolist(),
                                                             RobertaTokenizerFast.from_pretrained('microsoft/codebert-base'))
            print('Tokenizing msgs')
            msg_input_ids = DataPreprocessor.tokenize_messages(df['message'].tolist(),
                                                               GPT2TokenizerFast.from_pretrained('distilgpt2'))
            print('Constructing history')
            history = defaultdict(list)
            for msg, id in zip(msg_input_ids, df['author'].tolist()):
                history[id].append(msg)

            print('Saving history')
            with open(os.path.join(ds_root_path, f'{part}_history.json'), 'w') as outfile:
              json.dump(history, outfile)

            print('Saving data')
            if part != 'train':
              df['diff_input_ids'] = diff_input_ids
              df[['diff_input_ids', 'pos_in_history', 'author']].to_json(os.path.join(ds_root_path, f'{part}.json'), lines=True, orient='records')
            else:
              with open(os.path.join(ds_root_path, f'{part}.json'), 'w') as outfile:
                for diff_line, position_line, author_line in tqdm(zip(diff_input_ids,  df['pos_in_history'].tolist(),  df['author'].tolist())):
                  json.dump({'diff_input_ids': diff_line,
                             'pos_in_history': position_line,
                             'author': author_line}, outfile)
                  outfile.write('\n')