from dataclasses import dataclass

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

    Args:
        dataset_root: Directory with data, should contain files `train.jsonl`, `val.jsonl`, `test.jsonl`.
        preprocessor_chunksize: When data is preprocessed, how many examples should be in single chunk.
        testing: True to generate random numbers instead of actual data (used for tuning batch size).
        use_cache: True to look for preprocessed files, False to relaunch preprocessing even if preprocessed files are present.
        line_sep: Newline separator used in data (should be the same for diffs and messages).
        train_dataloader_conf: Configuration for train dataloader.
        val_dataloader_conf: Configuration for val dataloader.
        test_dataloader_conf: Configuration for test dataloader.
    """

    dataset_root: str = "raw_data/multilang"
    preprocessor_chunksize: int = 4096
    testing: bool = False
    use_cache: bool = False
    line_sep: str = "[NL]"
    train_dataloader_conf: DataLoaderConfig = DataLoaderConfig(batch_size=16, num_workers=4)
    val_dataloader_conf: DataLoaderConfig = DataLoaderConfig(batch_size=16, num_workers=4)
    test_dataloader_conf: DataLoaderConfig = DataLoaderConfig(batch_size=1, num_workers=1)
