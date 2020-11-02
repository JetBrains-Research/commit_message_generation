import os
import pprint
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, List

import torch

from dataset_utils.CommitMessageGenerationDataset import CommitMessageGenerationDataset
from models.EncoderDecoder import EncoderDecoder
from Config import Config, load_config
from experiments.AccuracyCalculation import AccuracyCalculation
from experiments.BleuCalculation import BleuCalculation


def measure_experiment_time(func) -> Any:
    start = time.time()
    ret = func()
    end = time.time()
    print(f'Duration: {str(timedelta(seconds=end - start))}')
    print()
    return ret


def save_predicted(max_top_k_predicted: List[List[List[str]]], dataset_name: str, config: Config) -> None:
    top_1_file_lines = []
    top_k_file_lines = []
    max_k = config['TOP_K'][-1]
    for predictions in max_top_k_predicted:
        top_1_file_lines.append("" if len(predictions) == 0 else ' '.join((str(i) for i in predictions[0])))
        top_k_file_lines.append('====NEW EXAMPLE====')
        for prediction in predictions[:max_k]:
            top_k_file_lines.append('' if len(prediction) == 0 else ' '.join(str(i) for i in prediction))

    top_1_path = os.path.join(config['OUTPUT_PATH'], f'{dataset_name}_predicted_top_1.txt')
    top_k_path = os.path.join(config['OUTPUT_PATH'], f'{dataset_name}_predicted_top_{max_k}.txt')
    with open(top_1_path, 'w') as top_1_file, open(top_k_path, 'w') as top_k_file:
        top_1_file.write('\n'.join(top_1_file_lines))
        top_k_file.write('\n'.join(top_k_file_lines))


def test_commit_message_generation_model(model: EncoderDecoder, train_size: int, val_size: int, test_size: int, config: Config, greedy: bool) -> None:
    train_dataset = CommitMessageGenerationDataset.load_data(os.path.join(config['DATASET_ROOT'], 'train'),
                                                             config, size=train_size)
    val_dataset = CommitMessageGenerationDataset.load_data(os.path.join(config['DATASET_ROOT'], 'val'),
                                                           config, size=val_size)
    test_dataset = CommitMessageGenerationDataset.load_data(os.path.join(config['DATASET_ROOT'], 'test'),
                                                            config, size=test_size)

    train_dataset_test_size = CommitMessageGenerationDataset.take_first_n_from_dataset(train_dataset, len(test_dataset))

    accuracy_calculation_experiment = AccuracyCalculation(model, max_len=100, greedy=greedy, config=config)
    bleu_calculation_experiment = BleuCalculation(config)

    model.eval()
    with torch.no_grad():
        test_max_top_k_predicted = measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(test_dataset, 'Test dataset')
        )
        save_predicted(test_max_top_k_predicted, dataset_name='test_dataset_commit_message_generator', config=config)
        measure_experiment_time(
            lambda: bleu_calculation_experiment.conduct(test_max_top_k_predicted, test_dataset,
                                                        'Test dataset')
        )

        val_max_top_k_predicted = measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(val_dataset, 'Validation dataset')
        )
        save_predicted(val_max_top_k_predicted, dataset_name='val_dataset_commit_message_generator', config=config)

        measure_experiment_time(
            lambda: bleu_calculation_experiment.conduct(val_max_top_k_predicted, val_dataset,
                                                        'Validation dataset')
        )
        train_max_top_k_predicted = measure_experiment_time(
            lambda: accuracy_calculation_experiment.conduct(train_dataset_test_size,
                                                            f'Train dataset (test size approximation)')
        )
        save_predicted(train_max_top_k_predicted, dataset_name='train_dataset_commit_message_generator', config=config)

        measure_experiment_time(
            lambda: bleu_calculation_experiment.conduct(train_max_top_k_predicted, train_dataset_test_size,
                                                        'Train dataset (test size approximation)')
        )


def print_results(results_root: str, config: Config) -> None:
    pprint.pprint(config.get_config())
    print('\n====STARTING COMMIT MSG GENERATOR EVALUATION====\n', end='')
    commit_msg_generator = torch.load(os.path.join(results_root, 'model_best_on_validation_commit_msg_generator.pt'),
                                      map_location=config['DEVICE'])
    test_commit_message_generation_model(commit_msg_generator, config)


def main() -> None:
    if len(sys.argv) != 3 and len(sys.argv) != 2:
        print("arguments: <results_root_dir> <is_test (optional, default false)>.")
    results_root_dir = sys.argv[1]
    is_test = len(sys.argv) > 2 and sys.argv[2] == 'test'
    config_path = Path(results_root_dir).joinpath('config.pkl')
    config = load_config(is_test, config_path)
    print_results(results_root_dir, config)


if __name__ == "__main__":
    main()
