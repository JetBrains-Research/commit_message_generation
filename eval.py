import os

import hydra
import nltk
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf

from src.data_utils import CMGDataModule
from src.model import EncoderDecoderModule, GPT2LMHeadModule
from src.utils import WandbOrganizer, prepare_cfg

nltk.download("omw-1.4")
nltk.download("wordnet")


@hydra.main(config_path="conf", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    pl.seed_everything(42)

    cfg = prepare_cfg(cfg)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    dm = CMGDataModule(
        **cfg.dataset,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=1,
    )
    dm.setup(stage=cfg.stage)

    if "wandb_logger" in cfg:
        wandb.Table.MAX_ROWS = 50000
        trainer_logger = pl.loggers.WandbLogger(
            name=f"context_ratio_{cfg.dataset.context_ratio}_{('with-history' if cfg.dataset.generate_with_history else 'without-history')}",
            project=cfg.wandb_logger.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            job_type="eval",
        )
    else:
        trainer_logger = False

    if "model_name" not in cfg or not cfg.model_name:
        cfg.model_name = WandbOrganizer.get_run_name(cfg.model, cfg.dataset)

    if "model_artifact" in cfg:
        assert isinstance(trainer_logger, pl.loggers.WandbLogger)
        artifact_name = f"{cfg.model_artifact.project}/{cfg.model_name}:latest"
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

    if "ckpt_path" in cfg:
        # initialize from fine-tuned checkpoint
        PATH = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)
        print("Checkpoint path\n", PATH, "\n")

        if cfg.model.encoder_decoder:
            # seq2seq model
            model = EncoderDecoderModule.load_from_checkpoint(
                PATH,
                **cfg.model,
                diff_tokenizer=dm._diff_tokenizer,
                msg_tokenizer=dm._msg_tokenizer,
                generation_kwargs=cfg.generation_kwargs,
                preds_artifact_name=f"{cfg.model_name}_preds",
                preds_artifact_type="multilang preds",
                preds_table_name=f"context_ratio_{cfg.dataset.context_ratio}_{('with-history' if cfg.dataset.generate_with_history else 'without-history')}",
            )
        else:
            # single decoder
            model = GPT2LMHeadModule.load_from_checkpoint(
                PATH,
                **cfg.model,
                tokenizer=dm._msg_tokenizer,
                generation_kwargs=cfg.generation_kwargs,
                preds_artifact_name=f"{cfg.model_name}_preds",
                preds_artifact_type="multilang preds",
                preds_table_name=f"context_ratio_{cfg.dataset.context_ratio}_{('with-history' if cfg.dataset.generate_with_history else 'without-history')}",
            )

        trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger)

        # -----------------------
        #          test         -
        # -----------------------
        trainer.test(ckpt_path=PATH, datamodule=dm, model=model)
    else:
        # use zero-shot pretrained model or even random model
        if cfg.model.encoder_decoder:
            # seq2seq model
            model = EncoderDecoderModule(
                **cfg.model,
                diff_tokenizer=dm._diff_tokenizer,
                msg_tokenizer=dm._msg_tokenizer,
                generation_kwargs=cfg.generation_kwargs,
                preds_artifact_name=f"{cfg.model_name}_preds",
                preds_artifact_type="multilang preds",
                preds_table_name=f"context_ratio_{cfg.context_ratxio}_{('with-history' if cfg.dataset.generate_with_history else 'without-history')}",
            )
        else:
            # single decoder
            model = GPT2LMHeadModule(
                **cfg.model,
                tokenizer=dm._msg_tokenizer,
                generation_kwargs=cfg.generation_kwargs,
                preds_artifact_name=f"{cfg.model_name}_preds",
                preds_artifact_type="multilang preds",
                preds_table_name=f"context_ratio_{cfg.dataset.context_ratio}_{('with-history' if cfg.dataset.generate_with_history else 'without-history')}",
            )

        trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger)

        # -----------------------
        #          test         -
        # -----------------------
        trainer.test(datamodule=dm, model=model)


if __name__ == "__main__":
    main()
