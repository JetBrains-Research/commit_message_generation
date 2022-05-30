import os

import hydra
import nltk
import pytorch_lightning as pl
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.data_utils import CMGDataModule
from src.model import EncoderDecoderModule, GPT2LMHeadModule

nltk.download("omw-1.4")
nltk.download("wordnet")


@hydra.main(config_path="conf", config_name="eval_config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    pl.seed_everything(42)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    wandb.Table.MAX_ROWS = 50000
    print(wandb.Table.MAX_ROWS)

    dm = CMGDataModule(
        **cfg.dataset,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=1,
    )
    dm.setup(stage=cfg.stage)

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True

    if "artifact" in cfg:
        assert isinstance(trainer_logger, pl.loggers.WandbLogger)

        trainer_logger.experiment.use_artifact(cfg.artifact.name).get_path(cfg.artifact.artifact_path).download(
            root=hydra.utils.to_absolute_path(cfg.artifact.local_path)
        )
        cfg.ckpt_path = os.path.join(hydra.utils.to_absolute_path(cfg.artifact.local_path), cfg.artifact.artifact_path)
    if "ckpt_path" in cfg:
        # initialize from already fine-tuned checkpo1int
        PATH = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)
        print("Checkpoint path\n", PATH, "\n")
        if cfg.model.encoder_decoder:
            # seq2seq model
            model = EncoderDecoderModule.load_from_checkpoint(
                PATH,
                diff_tokenizer=dm._diff_tokenizer,
                msg_tokenizer=dm._msg_tokenizer,
                wandb_artifact_name=cfg.model.wandb_artifact_name,
                wandb_table_name=cfg.model.wandb_table_name,
                generation_kwargs=cfg.generation_kwargs,
            )
        else:
            # single decoder
            model = GPT2LMHeadModule.load_from_checkpoint(
                PATH,
                tokenizer=dm._msg_tokenizer,
                wandb_artifact_name=cfg.model.wandb_artifact_name,
                wandb_table_name=cfg.model.wandb_table_name,
                generation_kwargs=cfg.generation_kwargs,
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
            )
        else:
            # single decoder
            model = GPT2LMHeadModule(
                **cfg.model,
                tokenizer=dm._msg_tokenizer,
                generation_kwargs=cfg.generation_kwargs,
            )

        trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
        trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger)

        # -----------------------
        #          test         -
        # -----------------------
        trainer.test(datamodule=dm, model=model)


if __name__ == "__main__":
    main()
