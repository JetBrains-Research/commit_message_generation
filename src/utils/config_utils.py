import logging

from omegaconf import DictConfig


def prepare_metrics_cfg(cfg: DictConfig) -> DictConfig:
    if not cfg.wandb.artifact.table_name:
        cfg.wandb.artifact.table_name = cfg.wandb.artifact.version
    return cfg


def prepare_dataset_cfg(dataset_cfg: DictConfig, model_dataset_cfg: DictConfig) -> DictConfig:
    if "preprocessor_conf" in model_dataset_cfg and "configuration" in model_dataset_cfg.preprocessor_conf:
        logging.info(f"{model_dataset_cfg.preprocessor_conf.configuration} preprocessing will be used")
        dataset_cfg.preprocessor_conf.configuration = model_dataset_cfg.preprocessor_conf.configuration

    for key in [
        "diff_tokenizer_name_or_path",
        "msg_tokenizer_name_or_path",
        "encoder_context_max_len",
        "decoder_context_max_len",
    ]:
        dataset_cfg[key] = model_dataset_cfg[key]

    return dataset_cfg
