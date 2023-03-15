from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING


@dataclass
class DataLoaderConfig:
    """
    DataLoader configuration.
    """

    batch_size: int = MISSING
    num_workers: int = MISSING


@dataclass
class DatasetConfig:
    """
    Basic data-related configuration.

    Attributes:
        dataset_root: Directory with data, should contain files `train.jsonl`, `val.jsonl`, `test.jsonl`.
        preprocessor_chunksize: When data is preprocessed, how many examples should be in single chunk.
        stage: Name of current stage, set to "sweep" to use correct logic for W&B sweep.
        add_history_to_inputs: True to save history for each input example,
         False to load history in RAM and build inputs on the fly.
        use_eval_downsample: True to use downsampled versions of validation and test sets.
        testing: True to generate random numbers instead of actual data (used for tuning batch size).
        use_cache: True to look for preprocessed files, False to relaunch preprocessing even if preprocessed files are present.
        line_sep: Newline separator used in data (should be the same for diffs and messages).
        train_dataloader_conf: Configuration for train dataloader.
        val_dataloader_conf: Configuration for val dataloader.
        test_dataloader_conf: Configuration for test dataloader.
    """

    dataset_root: str = "raw_data/multilang"
    preprocessor_chunksize: int = 4096
    stage: Optional[str] = None
    add_history_to_inputs: bool = True
    use_eval_downsample: bool = True
    testing: bool = False
    use_cache: bool = False
    line_sep: str = "\n"
    train_dataloader_conf: DataLoaderConfig = DataLoaderConfig(batch_size=16, num_workers=4)
    val_dataloader_conf: DataLoaderConfig = DataLoaderConfig(batch_size=16, num_workers=4)
    test_dataloader_conf: DataLoaderConfig = DataLoaderConfig(batch_size=1, num_workers=1)
