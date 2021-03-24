import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lr_logger_callback import LearningRateLogger

import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from model.encoder_decoder_module import EncoderDecoderModule
from dataset_utils.cmg_data_module import CMGDataModule


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    # -        init         -
    # -----------------------
    pl.seed_everything(42)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    # data
    dm = CMGDataModule(**cfg.dataset)
    dm.setup()

    # main module with model logic
    encoder_decoder = EncoderDecoderModule(**cfg.model,
                                           src_tokenizer=dm._src_tokenizer,
                                           trg_tokenizer=dm._trg_tokenizer,
                                           num_epochs=cfg.trainer.max_epochs,
                                           num_batches=len(dm.train_dataloader()))

    # logger
    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer_logger.watch(encoder_decoder, log='gradients', log_freq=250)

    # callbacks
    lr_logger = LearningRateLogger()

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_MRR_top5',
        mode='max'
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_MRR_top5',
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode='max'
    )

    # trainer
    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger, callbacks=[lr_logger,
                                                                          checkpoint_callback,
                                                                          early_stopping_callback])

    # -----------------------
    #         train         -
    # -----------------------
    trainer.fit(encoder_decoder, dm)
    # -----------------------
    #          test         -
    # -----------------------
    trainer.test(ckpt_path=checkpoint_callback.best_model_path, datamodule=dm)


if __name__ == '__main__':
    main()
