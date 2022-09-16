from omegaconf import DictConfig


def prepare_metrics_cfg(cfg: DictConfig) -> DictConfig:
    if not cfg.wandb.artifact.table_name:
        cfg.wandb.artifact.table_name = cfg.wandb.artifact.version
    return cfg
