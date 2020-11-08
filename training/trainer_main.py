from argparse import ArgumentParser

import pytorch_lightning as pl

from model.EncoderDecoderModule import EncoderDecoderModule
from dataset_utils.CMGDataModule import CMGDataModule

from Config import Config

# -----------------------
#          init         -
# -----------------------
config = Config()
dm = CMGDataModule(config)

parser = ArgumentParser()
# add model specific args
parser = EncoderDecoderModule.add_model_specific_args(parser, dm.config)
# add all the available trainer options to argparse
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()
dict_args = vars(args)
dict_args['tokenizer'] = dm.tokenizer
encoder_decoder = EncoderDecoderModule(**dict_args)
trainer = pl.Trainer.from_argparse_args(args)

# -----------------------
#         train         -
# -----------------------
trainer.fit(encoder_decoder, dm)
# -----------------------
#          test         -
# -----------------------
trainer.test(datamodule=dm)
