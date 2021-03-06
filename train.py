import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
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

    # freeze encoder and decoder
    for param in encoder_decoder.model.parameters():
        param.requires_grad = False
    # unfreeze cross-attention layers
    for layer in encoder_decoder.model.decoder.transformer.h:
        for param in layer.crossattention.parameters():
            param.requires_grad = True
    # unfreeze lm head
    for param in encoder_decoder.model.decoder.lm_head.parameters():
        param.requires_grad = True

    trainer_logger = instantiate(cfg.logger) if "logger" in cfg else True
    trainer_logger.watch(encoder_decoder, log='gradients', log_freq=250)
    lr_logger = LearningRateLogger()

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoint',
        save_top_k=1,
        verbose=True,
        monitor='val_MRR_top5',
        mode='max'
    )
    PATH = os.path.join(hydra.utils.get_original_cwd(), cfg.ckpt_path)
    trainer = pl.Trainer(**cfg.trainer, logger=trainer_logger, callbacks=[lr_logger, checkpoint_callback],
                         resume_from_checkpoint=PATH)
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
