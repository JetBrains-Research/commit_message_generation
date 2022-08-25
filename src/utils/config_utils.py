from omegaconf import DictConfig


def prepare_metrics_cfg(cfg: DictConfig) -> DictConfig:
    if not cfg.wandb.input_artifact_name:
        cfg.wandb.input_artifact_name = (
            f"{cfg.wandb.artifact_project}/{cfg.wandb.model_name}_preds:{cfg.wandb.model_config}"
        )
    if not cfg.wandb.input_table_name:
        cfg.wandb.input_table_name = cfg.wandb.model_config
    return cfg
