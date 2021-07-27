import os

import pytorch_lightning as pl

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.model import EncoderDecoderModule, GPT2LMHeadModule
from src.data_utils import CMGDataModule


@hydra.main(config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    pl.seed_everything(42)

    cfg.dataset.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.dataset.world_size = cfg.trainer.gpus if cfg.trainer.gpus > 0 else 1

    print("Local rank", cfg.dataset.local_rank)
    print("World size", cfg.dataset.world_size)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    dm = CMGDataModule(**cfg.dataset)
    dm.setup()

    if "ckpt_path" in cfg:
        # initialize from already fine-tuned checkpoint
        PATH = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)
        print("Checkpoint path\n", PATH, "\n")
        if cfg.model.encoder_decoder:
            # seq2seq model
            model = EncoderDecoderModule.load_from_checkpoint(PATH, num_gpus=1)
        else:
            # single decoder
            model = GPT2LMHeadModule.load_from_checkpoint(PATH)
        trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
        trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger)
        # -----------------------
        #          test         -
        # -----------------------
        trainer.test(ckpt_path=PATH, datamodule=dm, model=model)
    else:
        # initialize from pretrained weights or smth
        if cfg.model.encoder_decoder:
            # seq2seq model
            model = EncoderDecoderModule(
                **cfg.model,
                src_tokenizer=dm._src_tokenizer,
                trg_tokenizer=dm._trg_tokenizer,
                num_epochs=cfg.trainer.max_epochs,
                num_batches=dm.train._len // (cfg.dataset.train_dataloader_conf.batch_size * cfg.trainer.gpus),
                num_gpus=cfg.trainer.gpus if cfg.trainer.gpus > 0 else 1,
            )
        else:
            # single decoder
            model = GPT2LMHeadModule(**cfg.model, tokenizer=dm._trg_tokenizer)

        trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
        trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger)
        # -----------------------
        #          test         -
        # -----------------------
        trainer.test(datamodule=dm, model=model)


if __name__ == "__main__":
    main()
