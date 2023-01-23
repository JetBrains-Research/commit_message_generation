import logging
import os

import hydra
import nltk
import pytorch_lightning as pl
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.utilities.device_parser import num_cuda_devices
from wandb import Artifact

from src.data_utils import CMCDataModule
from src.model import CMCModule
from src.utils import WandbOrganizer, prepare_dataset_cfg

nltk.download("wordnet")


@hydra.main(config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    # -        init         -
    # -----------------------
    pl.seed_everything(42)

    world_size = None
    if cfg.trainer.accelerator == "cpu":
        world_size = 1
    elif cfg.trainer.accelerator == "gpu":
        if cfg.trainer.devices == "auto":
            world_size = num_cuda_devices()  # all available gpus
        elif isinstance(cfg.trainer.devices, int):
            if cfg.trainer.devices == -1:
                world_size = num_cuda_devices()  # all available gpus
            else:
                world_size = cfg.trainer.devices  # n first gpus
        elif isinstance(cfg.trainer.devices, str):
            if cfg.trainer.devices == "-1":
                world_size = num_cuda_devices()  # all available gpus
            else:
                world_size = len(cfg.trainer.devices.split(","))  # a list of specific gpus separated by ','
        elif isinstance(cfg.trainer.devices, ListConfig):
            world_size = len(cfg.trainer.devices)  # a list of specific gpus

    if world_size is None:
        raise ValueError("Unknown format for number of gpus")

    logging.info(f"Local rank: {int(os.environ.get('LOCAL_RANK', 0))}")
    logging.info(f"World size: {world_size}")

    cfg.dataset = prepare_dataset_cfg(cfg.dataset, model_dataset_cfg=cfg.model.dataset)
    dm = CMCDataModule(
        **cfg.dataset,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=world_size,
        shift_labels=cfg.model.model_configuration != "decoder",
        process_retrieved=cfg.model.model_configuration == "race",
    )

    dm.prepare_data()
    dm.setup(stage="fit")

    batch_size = cfg.dataset.train_dataloader_conf.batch_size * cfg.trainer.accumulate_grad_batches * world_size
    num_train_batches = dm.train.len // batch_size  # type: ignore[attr-defined]

    if "limit_train_batches" in cfg.trainer:
        num_train_batches = min(
            cfg.trainer.limit_train_batches // cfg.trainer.accumulate_grad_batches, num_train_batches
        )

    # main module with model logic
    model = CMCModule(
        **cfg.model,
        diff_tokenizer=dm.diff_tokenizer,
        msg_tokenizer=dm.msg_tokenizer,
        encoder_context_max_len=cfg.model.dataset.encoder_context_max_len,
        decoder_context_max_len=cfg.model.dataset.decoder_context_max_len,
        save_epoch=(cfg.trainer.max_epochs // 2) - 1,
        batch_size=batch_size,
        num_gpus=world_size,
        num_epochs=cfg.trainer.max_epochs,
        num_batches=num_train_batches,
    )

    cfg.model.learning_rate = model.learning_rate

    # logger
    if "wandb_logger" in cfg:
        use_wandb = True
        trainer_logger = pl.loggers.WandbLogger(
            name=WandbOrganizer.get_run_name(cfg.model, cfg.dataset),
            project=cfg.wandb_logger.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=WandbOrganizer.get_tags_train(cfg.model, cfg.dataset),
            job_type="train",
        )
        trainer_logger.watch(model, log="gradients", log_freq=250)
    else:
        use_wandb = False

    # callbacks
    lr_logger = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{WandbOrganizer.get_run_name(cfg.model, cfg.dataset)}_checkpoint",
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor="val_MRR_top5",
        mode="max",
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

    # trainer
    trainer = pl.Trainer(
        **cfg.trainer,
        logger=trainer_logger if use_wandb else True,
        callbacks=[lr_logger, checkpoint_callback, early_stopping_callback],
    )

    # -----------------------
    #         train         -
    # -----------------------
    trainer.fit(model, dm)

    # -----------------------
    #   save ckpt to wandb  -
    # -----------------------
    if (
        trainer_logger
        and isinstance(trainer_logger, pl.loggers.WandbLogger)
        and cfg.wandb_logger.save_model_as_artifact
    ):
        artifact = Artifact(
            name=WandbOrganizer.get_run_name(cfg.model, cfg.dataset),
            type="model",
            metadata={"tags": WandbOrganizer.get_tags_train(cfg.model, cfg.dataset)},
        )
        artifact.add_dir(f"{WandbOrganizer.get_run_name(cfg.model, cfg.dataset)}_checkpoint")
        trainer_logger.experiment.log_artifact(artifact)


if __name__ == "__main__":
    main()
