import os

import hydra
import nltk
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.cuda import device_count
from wandb import Artifact

from src.data_utils import CMGDataModule
from src.model import EncoderDecoderModule, GPT2LMHeadModule
from src.utils import LearningRateLogger

nltk.download("wordnet")


@hydra.main(config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    # -        init         -
    # -----------------------
    pl.seed_everything(42)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

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

    # logger
    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
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
    if "artifact" in cfg:
        assert isinstance(trainer_logger, pl.loggers.WandbLogger)
        artifact = Artifact(**cfg.artifact)
        artifact.add_dir("checkpoint")
        trainer_logger.experiment.log_artifact(artifact)


if __name__ == "__main__":
    main()
