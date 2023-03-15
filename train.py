import logging
import os

import hydra
import nltk
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.utilities.device_parser import num_cuda_devices
from wandb import Artifact

from conf import TrainConfig
from src.data_utils import CMCDataModule
from src.model import CMCModule
from src.utils import WandbOrganizer

nltk.download("wordnet")


@hydra.main(version_base="1.1", config_path="conf", config_name="train_config")
def main(cfg: TrainConfig) -> None:
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
        elif isinstance(cfg.trainer.devices, list):
            world_size = len(cfg.trainer.devices)  # a list of specific gpus

    if world_size is None:
        raise ValueError("Unknown format for number of gpus")

    logging.info(f"Local rank: {int(os.environ.get('LOCAL_RANK', 0))}")
    logging.info(f"World size: {world_size}")

    dm = CMCDataModule(
        dataset_cfg=cfg.dataset,
        model_cfg=cfg.model,
        input_cfg=cfg.input,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=world_size,
        shift_labels=cfg.model.configuration != "decoder",
        process_retrieved=cfg.model.configuration == "race",
    )

    dm.prepare_data()
    dm.setup(stage="fit")

    batch_size = cfg.dataset.train_dataloader_conf.batch_size * cfg.trainer.accumulate_grad_batches * world_size

    # main module with model logic
    model = CMCModule(
        model_cfg=cfg.model,
        diff_tokenizer=dm.diff_tokenizer,
        msg_tokenizer=dm.msg_tokenizer,
        learning_rate=cfg.optimizer.learning_rate,
        initial_batch_size=cfg.optimizer.initial_batch_size,
        weight_decay=cfg.optimizer.weight_decay,
        num_warmup_steps=cfg.optimizer.num_warmup_steps,
        ratio_warmup_steps=cfg.optimizer.ratio_warmup_steps,
        batch_size=batch_size,
    )
    cfg.optimizer.learning_rate = model.learning_rate

    run_name = WandbOrganizer.get_run_name(
        cfg.model,
        encoder_input_type=cfg.input.encoder_input_type,
        train_with_history=cfg.input.train_with_history,
    )
    run_tags = WandbOrganizer.get_tags_train(
        cfg.model,
        encoder_input_type=cfg.input.encoder_input_type,
        train_with_history=cfg.input.train_with_history,
    )

    # logger
    if cfg.logger.use_wandb:
        trainer_logger = pl.loggers.WandbLogger(
            name=run_name,
            project=cfg.logger.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=run_tags,
            job_type="train",
        )
        trainer_logger.watch(model, log="gradients", log_freq=250)

    # callbacks
    lr_logger = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{run_name}_checkpoint",
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor="val_MRR_top5",
        mode="max",
    )
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", verbose=True)

    # trainer
    trainer = pl.Trainer(
        **cfg.trainer,  # type: ignore[arg-type]
        logger=trainer_logger if cfg.logger.use_wandb else True,
        callbacks=[lr_logger, checkpoint_callback, early_stopping_callback],
    )

    # -----------------------
    #  zero-shot validation -
    # -----------------------
    trainer.validate(model, dm)

    # -----------------------
    #         train         -
    # -----------------------
    trainer.fit(model, dm)

    # -----------------------
    #   save ckpt to wandb  -
    # -----------------------
    if cfg.logger.use_wandb and cfg.logger.save_artifact:
        artifact = Artifact(
            name=run_name,
            type="model",
            metadata={"tags": run_tags},
        )
        artifact.add_dir(f"{run_name}_checkpoint")
        trainer_logger.experiment.log_artifact(artifact)


if __name__ == "__main__":
    main()
