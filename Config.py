import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch.backends.cudnn


class Config:
    _CONFIG = {
        'IS_TEST': False,
        'DATASET_ROOT': './data/CleanedJiang',
        'TOKENS_CODE_CHUNK_MAX_LEN': 100,  # TODO: drop inputs with more tokens?
        'MSG_MAX_LEN': 30,
        'OUTPUT_PATH': './experiments/last_execution/',
        'BLEU_PERL_SCRIPT_PATH': './experiments/multi-bleu.perl',  # Path to BLEU script calculator
        'LOWER': True,
        'LOWER_COMMIT_MSG': True,
        'SEED': 9382,
        'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'TOKEN_MIN_FREQ': 1,
        'LEARNING_RATE': 0.0001,
        'MAX_NUM_OF_EPOCHS': 2,
        'WORD_EMBEDDING_SIZE': 768,  # used in pretrained CodeBERT
        'ENCODER_HIDDEN_SIZE': 768,  # used in pretrained CodeBERT
        'DECODER_HIDDEN_SIZE': 256,
        'NUM_LAYERS': 2,
        'TEACHER_FORCING_RATIO': 0.9,
        'DROPOUT': 0.2,
        'USE_BRIDGE': True,
        'EARLY_STOPPING_ROUNDS': 80,
        'BEAM_SIZE': 10,
        'NUM_GROUPS': 1,
        'DIVERSITY_STRENGTH': None,
        'TOP_K': [1, 3, 5, 10, 50],
        'BEST_ON': 'BLEU',
        'START_BEST_FROM_EPOCH': 0,
        'LEAVE_ONLY_CHANGED': True,
        'VERBOSE': True,
        'BATCH_SIZE': 8,
        'TSNE_BATCH_SIZE': 1024,
        'VAL_BATCH_SIZE': 1,
        'TEST_BATCH_SIZE': 1,
        'SAVE_MODEL_EVERY': 10,
        'PRINT_EVERY_iTH_BATCH': 5,
        'MAKE_CUDA_REPRODUCIBLE': False,
    }

    _PATH_KEYS = ['DATASET_ROOT', 'OUTPUT_PATH', 'BLEU_PERL_SCRIPT_PATH']

    def __getitem__(self, key: str) -> Any:
        if key in self._PATH_KEYS:
            return os.path.abspath(os.path.join(os.path.dirname(__file__), self._CONFIG[key]))
        if isinstance(self._CONFIG[key], dict) and 'cmg' in self._CONFIG[key]:
            return self._CONFIG[key]['cmg']
        return self._CONFIG[key]

    def save(self) -> None:
        with open(os.path.join(self['OUTPUT_PATH'], 'config.pkl'), 'wb') as config_file:
            pickle.dump(self._CONFIG, config_file)

    def get_config(self) -> Dict[str, Any]:
        return self._CONFIG.copy()

    def change_config_for_test(self) -> None:
        self._CONFIG['IS_TEST'] = True
        self._CONFIG['DATASET_ROOT'] = \
            './data/CleanedJiang/'
        self._CONFIG['DATASET_ROOT_COMMIT'] = \
            './data/CleanedJiang/'
        self._CONFIG['MAX_NUM_OF_EPOCHS'] = {'cmg': 3}

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
