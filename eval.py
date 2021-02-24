import os

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

    PATH = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)

    print("Checkpoint path\n", PATH, '\n')

    encoder_decoder = EncoderDecoderModule.load_from_checkpoint(PATH)

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger)
    # -----------------------
    #          test         -
    # -----------------------
    trainer.test(ckpt_path=PATH, datamodule=dm, model=encoder_decoder)


if __name__ == '__main__':
    main()



