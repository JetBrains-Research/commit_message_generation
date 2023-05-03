import logging
import os
from typing import Any

import hydra
import nltk
import pytorch_lightning as pl
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.utilities.device_parser import num_cuda_devices

from conf import TrainConfig
from src.data_utils import CMCDataModule
from src.model import CMCModule
from src.utils import WandbOrganizer


def get_world_size(accelerator: str, devices: Any) -> int:
    """Determines world size for all possible ways of defining number of devices in Lightning.

    Args:
        accelerator: Argument for `pytorch_lightning.trainer`, corresponds to a device type.
        devices: Argument for `pytorch_lightning.trainer`, corresponds to a number of devices/specific devices to use.

    Returns:
        World size.
    """
    if accelerator == "cpu":
        return 1
    elif accelerator == "gpu":
        if devices == "auto":
            return num_cuda_devices()  # all available gpus
        elif isinstance(devices, int):
            if devices == -1:
                return num_cuda_devices()  # all available gpus
            else:
                return devices  # n first gpus
        elif isinstance(devices, str):
            if devices == "-1":
                return num_cuda_devices()  # all available gpus
            else:
                return len(devices.split(","))  # a list of specific gpus separated by ','
        elif isinstance(devices, list):
            return len(devices)  # a list of specific gpus

    raise ValueError("Unknown format for number of gpus")


@hydra.main(version_base="1.1", config_path="conf", config_name="train_config")
def main(cfg: TrainConfig) -> None:
    # -----------------------
    # -        init         -
    # -----------------------
    pl.seed_everything(42)

    # initializing gpus
    world_size = get_world_size(accelerator=cfg.trainer.accelerator, devices=cfg.trainer.devices)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    logging.info(f"Local rank: {local_rank}")
    logging.info(f"World size: {world_size}")

    dm = CMCDataModule(
        dataset_cfg=cfg.dataset,
        model_cfg=cfg.model,
        input_cfg=cfg.input,
        local_rank=local_rank,
        world_size=world_size,
        shift_labels=cfg.model.configuration != "decoder",
        process_retrieved=cfg.model.configuration == "race",
    )
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
        if cfg.logger.use_api_key:
            with open(hydra.utils.to_absolute_path("wandb_api_key.txt"), "r") as f:
                os.environ["WANDB_API_KEY"] = f.read().strip()
        trainer_logger = pl.loggers.WandbLogger(
            name=run_name,
            project=cfg.logger.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=run_tags,
            job_type="train",
        )

    if local_rank == 0:
        nltk.download("wordnet")

        if cfg.logger.use_wandb and cfg.model.configuration == "race":
            # download model checkpoint
            if cfg.logger.checkpoint.load_artifact:
                logging.info("Downloading fine-tuned CodeT5 checkpoint from W&B")
                artifact = wandb.use_artifact(
                    f"{cfg.logger.checkpoint.project}"
                    + "codet5"
                    + ("_with-history" if cfg.input.train_with_history else "_without-history")
                    + f":{cfg.logger.checkpoint.version}",
                )
                ckpt_path = os.path.join(
                    hydra.utils.to_absolute_path("artifacts"),
                    "codet5" + ("_with_history" if cfg.input.train_with_history else "_without_history"),
                )
                artifact.get_path(cfg.logger.checkpoint.artifact_path).download(root=ckpt_path)

            # download retrieved examples
            if cfg.logger.retrieval.load_artifact:
                logging.info("Downloading retrieved predictions from W&B")
                artifact = wandb.use_artifact(
                    f"{cfg.logger.retrieval.project}/"
                    + "codet5"
                    + ("_with-history" if cfg.input.train_with_history else "_without-history")
                    + "_retrieval"
                    + f":{cfg.logger.retrieval.version}",
                )

                for part in ["train", "val", "test"]:
                    artifact.get_path(f"{part}_predictions.jsonl").download(
                        root=os.path.join(
                            hydra.utils.to_absolute_path(dm.get_root_dir_for_part(cfg.dataset.dataset_root, part)),
                            "retrieval" + ("_with_history" if cfg.input.train_with_history else "_without_history"),
                        )
                    )
        dm.prepare_data(stage="fit")
    dm.setup(stage="fit")

    batch_size = cfg.dataset.train_dataloader_conf.batch_size * cfg.trainer.accumulate_grad_batches * world_size

    if cfg.logger.use_wandb and cfg.model.configuration == "race" and cfg.logger.checkpoint.load_artifact:
        transformers_ckpt_path = os.path.join(
            hydra.utils.to_absolute_path("artifacts"),
            "codet5" + ("_with_history" if cfg.input.train_with_history else "_without_history"),
            "transformers_format",
        )
        if local_rank == 0:
            model = CMCModule.load_from_checkpoint(
                os.path.join(
                    hydra.utils.to_absolute_path("artifacts"),
                    "codet5" + ("_with_history" if cfg.input.train_with_history else "_without_history"),
                    cfg.logger.checkpoint.artifact_path,
                ),
            )
            os.makedirs(transformers_ckpt_path, exist_ok=True)
            model.save_pretrained(transformers_ckpt_path)
        cfg.model.name_or_path = transformers_ckpt_path

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

    if isinstance(cfg.trainer.val_check_interval, float) and world_size > 1:
        logging.warning("Will divide `val_check_interval` by number of GPUs.")
        cfg.trainer.val_check_interval /= world_size

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
        artifact = wandb.Artifact(
            name=run_name,
            type="model",
            metadata={"tags": run_tags},
        )
        artifact.add_dir(f"{run_name}_checkpoint")
        wandb.log_artifact(artifact)


if __name__ == "__main__":
    main()
