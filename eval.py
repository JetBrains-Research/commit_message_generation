import logging
import os

import hydra
import nltk
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf

from src.data_utils import CMCDataModule
from src.model import CMCModule
from src.utils import WandbOrganizer

nltk.download("omw-1.4")
nltk.download("wordnet")


@hydra.main(config_path="conf", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    pl.seed_everything(42)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    if cfg.model.dataset.diff_tokenizer_name_or_path == cfg.model.dataset.msg_tokenizer_name_or_path:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dm = CMCDataModule(
        **cfg.dataset,
        **cfg.model.dataset,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=1,
        shift_labels=cfg.model.model_configuration == "encoder_decoder",
    )
    dm.setup(stage=cfg.stage)

    if "wandb_logger" in cfg:
        wandb.Table.MAX_ROWS = 50000
        use_wandb = True
        trainer_logger = pl.loggers.WandbLogger(
            name=f"context_ratio_{cfg.dataset.context_ratio}_{('with-history' if cfg.dataset.generate_with_history else 'without-history')}",
            project=cfg.wandb_logger.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            job_type="eval",
        )
    else:
        use_wandb = False

    if "model_name" not in cfg or not cfg.model_name:
        cfg.model_name = WandbOrganizer.get_run_name(cfg.model, cfg.dataset)

    if "model_artifact" in cfg:
        assert isinstance(trainer_logger, pl.loggers.WandbLogger)
        artifact_name = f"{cfg.model_artifact.project}/{cfg.model_name}:{cfg.model_artifact.version}"
        artifact = trainer_logger.experiment.use_artifact(artifact_name)
        if "tags" in artifact.metadata:
            trainer_logger.experiment.tags = artifact.metadata["tags"] + WandbOrganizer.get_tags_generate(cfg.dataset)

        artifact.get_path(cfg.model_artifact.artifact_path).download(
            root=hydra.utils.to_absolute_path(f"{cfg.model_artifact.local_path}/{cfg.model_name}")
        )

        cfg.ckpt_path = os.path.join(
            hydra.utils.to_absolute_path(f"{cfg.model_artifact.local_path}/{cfg.model_name}"),
            cfg.model_artifact.artifact_path,
        )

    preds_table_tags = [f"context-ratio_{cfg.dataset.context_ratio}"]
    if cfg.model.model_configuration == "encoder_decoder" and cfg.dataset.encoder_input_type == "diff":
        if cfg.dataset.generate_with_history:
            preds_table_tags.append("with-history")
        else:
            preds_table_tags.append("without-history")
    preds_table_name = "_".join(preds_table_tags)

    if "ckpt_path" in cfg:
        # initialize from fine-tuned checkpoint
        PATH = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)
        print("Checkpoint path\n", PATH, "\n")

        model = CMCModule.load_from_checkpoint(
            PATH,
            **cfg.model,
            diff_tokenizer=dm._diff_tokenizer,
            msg_tokenizer=dm._msg_tokenizer,
            generation_kwargs=cfg.generation_kwargs,
            preds_artifact_name=f"{cfg.model_name}_preds",
            preds_artifact_type="multilang preds",
            preds_table_name=preds_table_name,
        )
    else:
        logging.info("Training from scratch")
        # use zero-shot pretrained model or even random model
        model = CMCModule(
            **cfg.model,
            diff_tokenizer=dm._diff_tokenizer,
            msg_tokenizer=dm._msg_tokenizer,
            generation_kwargs=cfg.generation_kwargs,
            preds_artifact_name=f"{cfg.model_name}_preds",
            preds_artifact_type="multilang preds",
            preds_table_name=preds_table_name,
        )

    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger if use_wandb else True)

    # -----------------------
    #          test         -
    # -----------------------
    trainer.test(datamodule=dm, model=model)


if __name__ == "__main__":
    main()
