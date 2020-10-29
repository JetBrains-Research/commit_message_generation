import subprocess
import tempfile
from typing import List, Tuple
from torch.utils.data import Dataset
from Config import Config


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


class BleuCalculation:
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    def get_bleu_script_output(self, pred_str, trg_str: str) -> Tuple[str, str]:
        print("BLEU top 1 predictions", pred_str)
        print("BLEU targets", trg_str)
        with tempfile.NamedTemporaryFile(mode='w') as file_with_targets:
            file_with_targets.write(trg_str + '\n')
            file_with_targets.flush()
            process = subprocess.Popen([self.config['BLEU_PERL_SCRIPT_PATH'], file_with_targets.name],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result = process.communicate(input=(pred_str+'\n').encode())
            return result

    def conduct(self, predictions: List[List[str]], dataset, dataset_label: str) -> None:
        print(f'Start conducting BLEU calculation experiment for {dataset_label}...')
        trg_str = self.preprocess_dataset_for_bleu(dataset)
        pred_str = self.preprocess_predictions_for_bleu(predictions)
        result = self.get_bleu_script_output(pred_str, trg_str)
        print(result[0])
        print(f'Errors: {result[1]}')

    def get_bleu_score(self, predictions: List[List[List[str]]], dataset: Dataset) -> float:
        trg_str = self.preprocess_dataset_for_bleu(dataset)
        pred_str = self.preprocess_predictions_for_bleu(predictions)
        result = self.get_bleu_script_output(pred_str, trg_str)
        print(result[0])
        words = result[0].split()
        if len(words) > 2 and len(words[2]) > 0 and isfloat(words[2][:-1]):
            bleu_score = float(result[0].split()[2][:-1])
            return bleu_score
        else:
            print('Warning: something wrong with bleu score')
            print(f'Errors: {result[1]}')
            return 0

    @staticmethod
    def preprocess_dataset_for_bleu(dataset: Dataset, pad_token_id=1) -> str:
        data_iterator = DataLoader(dataset, batch_size=len(dataset))  # load whole dataset in one batch
        for batch in data_iterator:
            targets = batch['target']['input_ids'].tolist()
            # remove trailing pad tokens ids
            for i, el in enumerate(targets):
                try:
                    ind = el.index(pad_token_id)
                except ValueError:  # no padding
                    ind = len(el)
                targets[i] = el[:ind]
            # separate elements in each row with spaces
            # separate rows with newline \n
            return '\n'.join([' '.join([str(i) for i in lst]) for lst in targets])

    @staticmethod
    def preprocess_predictions_for_bleu(predictions: List[List[List[str]]]) -> str:
        top_1_predictions = ['' if len(prediction) == 0 else ' '.join([str(i) for i in prediction[0]]) for prediction in
                             predictions]
        return '\n'.join(top_1_predictions)