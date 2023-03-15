import logging
import os

import hydra
import nltk
import pytorch_lightning as pl
import wandb
from omegaconf import OmegaConf

from conf import EvalConfig
from src.data_utils import CMCDataModule
from src.model import CMCModule
from src.utils import WandbOrganizer

nltk.download("omw-1.4")
nltk.download("wordnet")


@hydra.main(version_base="1.1", config_path="conf", config_name="eval_config")
def main(cfg: EvalConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    pl.seed_everything(42)

    if cfg.model.diff_tokenizer_name_or_path == cfg.model.msg_tokenizer_name_or_path:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dm = CMCDataModule(
        dataset_cfg=cfg.dataset,
        model_cfg=cfg.model,
        input_cfg=cfg.input,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=1,
        shift_labels=cfg.model.configuration != "decoder",
        process_retrieved=cfg.model.configuration == "race",
    )
    dm.prepare_data()
    dm.setup(stage=cfg.stage)

    if cfg.logger.use_wandb:
        wandb.Table.MAX_ROWS = 50000
        trainer_logger = pl.loggers.WandbLogger(
            name=f"context_ratio_{cfg.input.context_ratio}_{('with-history' if cfg.input.generate_with_history else 'without-history')}",
            project=cfg.logger.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            job_type="eval",
        )

    run_name = WandbOrganizer.get_run_name(
        cfg.model,
        encoder_input_type=cfg.input.encoder_input_type,
        train_with_history=cfg.input.train_with_history,
    )

    if cfg.logger.use_wandb and cfg.logger.load_artifact:
        artifact_name = f"{cfg.logger.artifact_config.project}/{run_name}:{cfg.logger.artifact_config.version}"
        artifact = trainer_logger.experiment.use_artifact(artifact_name)
        if "tags" in artifact.metadata:
            trainer_logger.experiment.tags = artifact.metadata["tags"] + WandbOrganizer.get_tags_generate(
                generate_with_history=cfg.input.generate_with_history, context_ratio=cfg.input.context_ratio
            )

        artifact.get_path(cfg.logger.artifact_config.artifact_path).download(
            root=hydra.utils.to_absolute_path(f"{cfg.logger.artifact_config.local_path}/{run_name}")
        )

        cfg.ckpt_path = os.path.join(
            hydra.utils.to_absolute_path(f"{cfg.logger.artifact_config.local_path}/{run_name}"),
            cfg.logger.artifact_config.artifact_path,
        )

    preds_table_tags = [f"context-ratio_{cfg.input.context_ratio}"]
    if cfg.input.encoder_input_type == "diff":
        if cfg.input.generate_with_history:
            preds_table_tags.append("with-history")
        else:
            preds_table_tags.append("without-history")
    preds_table_name = "_".join(preds_table_tags)

    if cfg.ckpt_path:
        # initialize from fine-tuned checkpoint
        PATH = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)
        print("Checkpoint path\n", PATH, "\n")

        model = CMCModule.load_from_checkpoint(
            PATH,
            model_cfg=cfg.model,
            diff_tokenizer=dm.diff_tokenizer,
            msg_tokenizer=dm.msg_tokenizer,
            generation_kwargs=cfg.generation,  # type: ignore[arg-type]
            preds_artifact_name=f"{run_name}_preds",
            preds_artifact_type="multilang preds",
            preds_table_name=preds_table_name,
        )
    else:
        logging.info("Using zero-shot model")
        # use zero-shot pretrained model or even random model
        model = CMCModule(
            model_cfg=cfg.model,
            diff_tokenizer=dm.diff_tokenizer,
            msg_tokenizer=dm.msg_tokenizer,
            generation_kwargs=cfg.generation,  # type: ignore[arg-type]
            preds_artifact_name=f"{run_name}_preds",
            preds_artifact_type="multilang preds",
            preds_table_name=preds_table_name,
        )

    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger if cfg.logger.use_wandb else True)  # type: ignore[arg-type]

    # -----------------------
    #          test         -
    # -----------------------
    trainer.test(datamodule=dm, model=model)


if __name__ == "__main__":
    main()
