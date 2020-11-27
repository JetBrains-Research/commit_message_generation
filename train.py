import pytorch_lightning as pl

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

    encoder_decoder = EncoderDecoderModule(**cfg.model, tokenizer=dm._tokenizer)

    # freee codebert
    for param in encoder_decoder.encoder.parameters():
        param.requires_grad = False

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer_logger.watch(encoder_decoder, log='gradients', log_freq=250)

    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger)
    # -----------------------
    #       tune lr         -
    # -----------------------
    trainer.tune(model=encoder_decoder, datamodule=dm)
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