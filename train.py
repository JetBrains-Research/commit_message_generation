import os

import hydra
import nltk
import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.cuda import device_count
from wandb import Artifact

from src.data_utils import CMGDataModule
from src.model import EncoderDecoderModule, GPT2LMHeadModule
from src.utils import LearningRateLogger, WandbOrganizer, prepare_cfg

nltk.download("wordnet")


@hydra.main(config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    # -        init         -
    # -----------------------
    pl.seed_everything(42)

    cfg = prepare_cfg(cfg)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    if "gpus" not in cfg.trainer:
        world_size = 1  # cpu
    elif isinstance(cfg.trainer.gpus, int):
        if cfg.trainer.gpus == -1:
            world_size = device_count()  # all available gpus
        elif cfg.trainer.gpus == 0:
            world_size = 1  # cpu
        else:
            world_size = cfg.trainer.gpus  # n first gpus
    elif isinstance(cfg.trainer.gpus, str):
        if cfg.trainer.gpus == "-1":
            world_size = device_count()  # all available gpus
        else:
            world_size = len(cfg.trainer.gpus.split(","))  # a list of specific gpus separated by ','
    elif isinstance(cfg.trainer.gpus, ListConfig):
        world_size = len(cfg.trainer.gpus)  # a list of specific gpus
    else:
        raise ValueError("Unknown format for number of gpus")

    print(f"Local rank: {int(os.environ.get('LOCAL_RANK', 0))}")
    print(f"World size: {world_size}")

    dm = CMGDataModule(
        **cfg.dataset,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=world_size,
    )
    dm.setup(stage="fit")

    # logger
    if "wandb_logger" in cfg:
        trainer_logger = pl.loggers.WandbLogger(
            name=WandbOrganizer.get_run_name(cfg.model, cfg.dataset),
            project=cfg.wandb_logger.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=WandbOrganizer.get_tags_train(cfg.model, cfg.dataset),
            job_type="train",
        )
    else:
        trainer_logger = False

    # main module with model logic
    if cfg.model.encoder_decoder:
        # seq2seq model
        model = EncoderDecoderModule(
            **cfg.model,
            diff_tokenizer=dm._diff_tokenizer,
            msg_tokenizer=dm._msg_tokenizer,
            num_epochs=cfg.trainer.max_epochs,
            num_batches=dm.train._len // (cfg.dataset.train_dataloader_conf.batch_size * world_size),
            num_gpus=world_size,
        )
    else:
        # decoder
        model = GPT2LMHeadModule(
            **cfg.model,
            tokenizer=dm._msg_tokenizer,
            num_epochs=cfg.trainer.max_epochs,
            num_batches=dm.train._len // (cfg.dataset.train_dataloader_conf.batch_size * world_size),
            num_gpus=world_size,
        )
    if trainer_logger:
        trainer_logger.watch(model, log="gradients", log_freq=250)

    # callbacks
    lr_logger = LearningRateLogger()
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoint", save_top_k=1, save_last=True, verbose=True, monitor="val_MRR_top5", mode="max"
    )

    # trainer
    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger, callbacks=[lr_logger, checkpoint_callback])

    # -----------------------
    #         train         -
    # -----------------------
    trainer.fit(model, dm)

    # -----------------------
    #   save ckpt to wandb  -
    # -----------------------
    if "wandb_logger" in cfg and cfg.wandb_logger.save_model_as_artifact:
        assert isinstance(trainer_logger, pl.loggers.WandbLogger)
        artifact = Artifact(
            name=WandbOrganizer.get_run_name(cfg.model, cfg.dataset),
            type="multilang model",
            metadata={"tags": WandbOrganizer.get_tags_train(cfg.model, cfg.dataset)},
        )
        artifact.add_dir("checkpoint")
        trainer_logger.experiment.log_artifact(artifact)


if __name__ == "__main__":
    main()
