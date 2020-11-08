import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch.backends.cudnn


class Config:
    _CONFIG = {
        'MODEL_NAME_OR_PATH': 'microsoft/codebert-base',

        'DATASET_ROOT': './raw_data/CleanedJiang',
        'OUTPUT_PATH': './experiments/last_execution/',
        'BLEU_PERL_SCRIPT_PATH': './experiments/multi-bleu.perl',  # Path to BLEU script calculator

        'DIFF_MAX_LEN': 100,
        'MSG_MAX_LEN': 30,

        'EMBEDDING_SIZE': 768,
        'HIDDEN_SIZE_ENCODER': 768,  # used in pretrained CodeBERT
        'HIDDEN_SIZE_DECODER': 768,
        'NUM_LAYERS': 2,
        'NUM_HEADS': 8,
        'TEACHER_FORCING_RATIO': 0.75,
        'DROPOUT': 0.2,
        'USE_BRIDGE': True,

        'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'LEARNING_RATE': 0.0001,
        'START_BEST_FROM_EPOCH': 0,
        'MAX_EPOCHS': 10,
        'SAVE_MODEL_EVERY': 10,
        'PRINT_EVERY_iTH_BATCH': 5,
        'TRAIN_BATCH_SIZE': 8,
        'VAL_BATCH_SIZE': 8,
        'TEST_BATCH_SIZE': 8,

        'TOP_K': [1, 3, 5, 10],
        'BEST_ON': 'BLEU',

        'VERBOSE': True,
        'MAKE_CUDA_REPRODUCIBLE': False,
        'SEED': 9382,
    }

    _PATH_KEYS = ['DATASET_ROOT', 'OUTPUT_PATH', 'BLEU_PERL_SCRIPT_PATH']

    def __getitem__(self, key: str) -> Any:
        if key in self._PATH_KEYS:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), self._CONFIG[key]))
        return self._CONFIG[key]

    def save(self) -> None:
        with open(os.path.join(self['OUTPUT_PATH'], 'config.pkl'), 'wb') as config_file:
            pickle.dump(self._CONFIG, config_file)

    def get_config(self) -> Dict[str, Any]:
        return self._CONFIG.copy()

    @property
    def CONFIG(self):
        return self._CONFIG


def load_config(is_test: bool, path_to_config: Path = None) -> Config:
    config = Config()
    if path_to_config is not None:
        with path_to_config.open(mode='rb') as config_file:
            config._CONFIG = pickle.load(config_file)
            config._CONFIG['DEVICE'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if config['SEED'] is not None:
        make_reproducible(config['SEED'], config['MAKE_CUDA_REPRODUCIBLE'])
    if is_test:
        config.change_config_for_test()
    return config


def make_reproducible(seed: int, make_cuda_reproducible: bool) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if make_cuda_reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False