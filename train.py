import pytorch_lightning as pl
from lr_logger_callback import LearningRateLogger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from model.encoder_decoder_module import EncoderDecoderModule
from dataset_utils.cmg_data_module import CMGDataModule


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    pl.seed_everything(42)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    dm = CMGDataModule(**cfg.dataset)
    dm.setup()

    encoder_decoder = EncoderDecoderModule(**cfg.model,
                                           src_tokenizer=dm._src_tokenizer,
                                           trg_tokenizer=dm._trg_tokenizer,
                                           num_epochs=cfg.trainer.max_epochs,
                                           num_batches=len(dm.train_dataloader()) + len(dm.val_dataloader()))

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer_logger.watch(encoder_decoder, log='gradients', log_freq=250)
    lr_logger = LearningRateLogger()

    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger, callbacks=[lr_logger])
    # -----------------------
    #         train         -
    # -----------------------
    trainer.fit(encoder_decoder, dm)
    # -----------------------
    #          test         -
    # -----------------------
    trainer.test(datamodule=dm)


if __name__ == '__main__':
    main()