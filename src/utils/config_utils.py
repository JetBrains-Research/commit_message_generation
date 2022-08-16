from omegaconf import DictConfig


def prepare_cfg(cfg: DictConfig) -> DictConfig:
    for key in ["dataset_root", "diff_tokenizer_name_or_path", "msg_tokenizer_name_or_path"]:
        if not cfg.dataset[key]:
            if key == "dataset_root" and key not in cfg.model:
                raise ValueError(f"Please make sure to set `dataset_root` field either in dataset or model config")
            cfg.dataset[key] = cfg.model[key]
    return cfg
