import os

import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.model import EncoderDecoderModule, GPT2LMHeadModule
from src.dataset_utils import CMGDataModule


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    pl.seed_everything(42)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    dm = CMGDataModule(**cfg.dataset, actual_generation=cfg.actual_generation)

    if cfg.model.encoder_decoder:
        if "ckpt_path" in cfg:
            PATH = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)
            print("Checkpoint path\n", PATH, "\n")
            model = EncoderDecoderModule.load_from_checkpoint(
                PATH,
                context_ratio=int(cfg.dataset.context_ratio * 100),
                actual_generation=cfg.actual_generation,
                num_gpus=1,
            )
        else:
            model = EncoderDecoderModule(
                **cfg.model,
                context_ratio=int(cfg.dataset.context_ratio * 100),
                num_gpus=1,
                diff_tokenizer=dm._src_tokenizer,
                msg_tokenizer=dm._trg_tokenizer,
            )
    else:
        if "ckpt_path" in cfg:
            PATH = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)
            print("Checkpoint path\n", PATH, "\n")

            model = GPT2LMHeadModule.load_from_checkpoint(
                PATH, context_ratio=int(cfg.dataset.context_ratio * 100), actual_generation=cfg.actual_generation
            )
        else:
            model = GPT2LMHeadModule(
                context_ratio=int(cfg.dataset.context_ratio * 100),
                decoder_name_or_path=cfg.model.decoder_name_or_path,
                actual_generation=cfg.actual_generation,
                tokenizer=dm._trg_tokenizer,
            )

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger)
    # -----------------------
    #          test         -
    # -----------------------
    trainer.test(datamodule=dm, model=model)


if __name__ == "__main__":
    main()
