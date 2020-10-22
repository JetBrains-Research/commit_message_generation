import subprocess
import tempfile
from typing import List, Tuple

from Config import Config

# TODO: test BleuCalculation


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

    def get_bleu_script_output(self, predictions, dataset) -> Tuple[str, str]:
        top_1_predictions = ['' if len(prediction) == 0 else ' '.join(prediction[0]) for prediction in predictions]
        targets = [' '.join(example) for example in dataset]
        with tempfile.NamedTemporaryFile(mode='w') as file_with_targets:
            file_with_targets.write('\n'.join(targets))
            file_with_targets.flush()
            process = subprocess.Popen([self.config['BLEU_PERL_SCRIPT_PATH'], file_with_targets.name],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            result = process.communicate(input=('\n'.join(top_1_predictions)).encode())
            return result

    def conduct(self, predictions: List[List[str]], dataset, dataset_label: str) -> None:
        print(f'Start conducting BLEU calculation experiment for {dataset_label}...')
        result = self.get_bleu_script_output(predictions, dataset)
        print(result[0])
        print(f'Errors: {result[1]}')

    def get_bleu_score(self, predictions: List[List[List[str]]], dataset) -> float:
        result = self.get_bleu_script_output(predictions, dataset)
        print(result[0])
        words = result[0].split()
        if len(words) > 2 and len(words[2]) > 0 and isfloat(words[2][:-1]):
            bleu_score = float(result[0].split()[2][:-1])
            return bleu_score
        else:
            print('Warning: something wrong with bleu score')
            print(f'Errors: {result[1]}')
            return 0