import os

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data_utils import CMGDataModule
from src.model import EncoderDecoderModule, GPT2LMHeadModule
from src.utils import LearningRateLogger


@hydra.main(config_path="conf", config_name="train_config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    # -        init         -
    # -----------------------
    pl.seed_everything(42)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    dm = CMGDataModule(
        **cfg.dataset,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=cfg.trainer.gpus if cfg.trainer.gpus > 0 else 1,
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
            num_batches=dm.train._len // (cfg.dataset.train_dataloader_conf.batch_size * cfg.trainer.gpus),
            num_gpus=cfg.trainer.gpus if cfg.trainer.gpus > 0 else 1,
        )
    else:
        # decoder
        model = GPT2LMHeadModule(
            **cfg.model,
            tokenizer=dm._msg_tokenizer,
            num_epochs=cfg.trainer.max_epochs,
            num_batches=dm.train._len // (cfg.dataset.train_dataloader_conf.batch_size * cfg.trainer.gpus),
            num_gpus=cfg.trainer.gpus if cfg.trainer.gpus > 0 else 1,
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


if __name__ == "__main__":
    main()
