import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lr_logger_callback import LearningRateLogger

import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from model.encoder_decoder_module import EncoderDecoderModule
from model.gpt2_lm_head_module import GPT2LMHeadModule
from dataset_utils.cmg_data_module import CMGDataModule


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    # -        init         -
    # -----------------------
    pl.seed_everything(42)

    cfg.dataset.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.dataset.world_size = cfg.trainer.gpus if cfg.trainer.gpus > 0 else 1

    print('Local rank', cfg.dataset.local_rank)
    print('World size', cfg.dataset.world_size)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")
    # data
    dm = CMGDataModule(**cfg.dataset)
    dm.setup()

    # main module with model logic
    if cfg.model.encoder_decoder:
        # seq2seq model
        model = EncoderDecoderModule(**cfg.model,
                                     src_tokenizer=dm._src_tokenizer,
                                     trg_tokenizer=dm._trg_tokenizer,
                                     num_epochs=cfg.trainer.max_epochs,
                                     num_batches=dm.train._len // (
                                             cfg.dataset.train_dataloader_conf.batch_size * cfg.trainer.gpus),
                                     num_gpus=cfg.trainer.gpus if cfg.trainer.gpus > 0 else 1)
    else:
        # single decoder
        model = GPT2LMHeadModule(**cfg.model,
                                 tokenizer=dm._trg_tokenizer,
                                 num_epochs=cfg.trainer.max_epochs,
                                 num_batches=dm.train._len // (
                                         cfg.dataset.train_dataloader_conf.batch_size * cfg.trainer.gpus),
                                 num_gpus=cfg.trainer.gpus if cfg.trainer.gpus > 0 else 1)

    # logger
    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer_logger.watch(model, log='gradients', log_freq=250)

    # callbacks
    lr_logger = LearningRateLogger()

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoint',
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='val_MRR_top5',
        mode='max'
    )

    # trainer
    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger, callbacks=[lr_logger,
                                                                          checkpoint_callback])

    # -----------------------
    #         train         -
    # -----------------------
    trainer.fit(model, dm)
    # -----------------------
    #          test         -
    # -----------------------
    trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    main()
