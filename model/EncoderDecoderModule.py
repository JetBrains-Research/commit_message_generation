from argparse import ArgumentParser

import pytorch_lightning as pl

import torch
import torch.nn.functional as F

from transformers import RobertaConfig, RobertaModel, AdamW

from model.Decoder import Decoder

from metrics.MyAccuracyMetric import MyAccuracyMetric
from metrics.MyBleuMetric import MyBleuMetric


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self, embedding_dim, vocab_size, hidden_size_decoder, hidden_size_encoder,
                 num_heads, num_layers, dropout, bridge, teacher_forcing_ratio, learning_rate,
                 max_len, pad_token_id, tokenizer, model_name_or_path, **kwargs):
        super().__init__()

        self.tokenizer = tokenizer
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.pad_token_id = pad_token_id

        self.encoder_config = RobertaConfig.from_pretrained(model_name_or_path)
        self.encoder = RobertaModel.from_pretrained(model_name_or_path, config=self.encoder_config)

        self.decoder = Decoder(embed_dim=embedding_dim,
                               vocab_size=vocab_size,
                               hidden_size=hidden_size_decoder,
                               hidden_size_encoder=hidden_size_encoder,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               dropout=dropout,
                               bridge=bridge,
                               teacher_forcing_ratio=teacher_forcing_ratio)

        self.accuracy = MyAccuracyMetric(self.pad_token_id)
        self.bleu = MyBleuMetric()

    def forward(self, batch):
        src, trg = batch
        # encode step
        encoder_output, _, encoder_final = self.encoder(input_ids=src['input_ids'],
                                                        attention_mask=src['attention_mask'],
                                                        output_hidden_states=True)
        t = encoder_final[0].shape[1] - 1
        encoder_final = torch.stack(encoder_final)[:, :, t, :][-self.decoder.num_layers:, :]
        # decode step
        return self.decoder(trg['input_ids'],
                            trg['attention_mask'],
                            encoder_output, encoder_final,
                            torch.logical_not(src['attention_mask']))

    def training_step(self, batch, batch_idx):
        src, trg = batch
        decoder_states, hidden, output = self(batch)

        train_loss = F.nll_loss(output.view(-1, output.size(-1)), trg['input_ids'].view(-1),
                                reduction='mean', ignore_index=self.pad_token_id)

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        src, trg = batch
        decoder_states, hidden, output = self(batch)

        val_loss = F.nll_loss(output.view(-1, output.size(-1)), trg['input_ids'].view(-1),
                              reduction='mean', ignore_index=self.pad_token_id)

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch):
        # TODO: implement beam search and add choice between beam search/greedy approaches
        src, trg = batch
        encoder_output, _, encoder_final = self.encoder(input_ids=src['input_ids'],
                                                        attention_mask=src['attention_mask'],
                                                        output_hidden_states=True)
        t = encoder_final[0].shape[1] - 1
        encoder_final = torch.stack(encoder_final)[:, :, t, :][-self.decoder.num_layers:, :]

        prev_y = torch.ones(len(trg), 1)
        prev_y_mask = torch.ones_like(prev_y)

        preds = torch.zeros((batch['input_ids'].shape[0], self.hparams.max_len))
        hidden = None

        for i in range(self.hparams.max_len):
            decoder_states, hidden, output = self.decoder(prev_y,
                                                          prev_y_mask,
                                                          encoder_output, encoder_final,
                                                          torch.logical_not(src['attention_mask'], hidden=hidden))
            _, next_word = torch.max(output, dim=2)
            preds[:, i] = next_word
            prev_y[:, 0] = next_word  # change prev id to generated id
        preds = preds.detach().cpu()
        return {'preds': preds, 'targets': trg['input_ids']}

    def on_test_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()

        # TODO: should we compute accuracy on decoded strings or with token ids (probably faster)?
        # compute accuracy with tensors
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu()
        targets = torch.cat([x['targets'] for x in outputs]).detach().cpu()

        acc = self.accuracy.compute(preds, targets)

        # compute BLEU with decoded strings
        targets = [
            self.tokenizer.decode(example, skip_special_tokens=True, clean_up_tokenization_spaces=False).split(' ')
            for example in targets.tolist()]

        preds = [self.tokenizer.decode(example, skip_special_tokens=True, clean_up_tokenization_spaces=False).split(' ')
                 for example in preds.tolist()]

        bleu = self.bleu.compute(translate_corpus=preds, reference_corpus=targets)

        self.log('val_accuracy', acc, prog_bar=True, logger=True)
        self.log('val_bleu', bleu, prog_bar=True, logger=True)
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser, config):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model_name_or_path", default=config['MODEL_NAME_OR_PATH'], type=str)
        parser.add_argument("--embedding_dim", default=config['EMBEDDING_SIZE'], type=int)
        parser.add_argument("--vocab_size", default=config['VOCAB_SIZE'], type=int)
        parser.add_argument("--hidden_size_decoder", default=config['HIDDEN_SIZE_DECODER'], type=int)
        parser.add_argument("--hidden_size_encoder", default=config['HIDDEN_SIZE_ENCODER'], type=int)
        parser.add_argument("--num_heads", default=config['NUM_HEADS'], type=int)
        parser.add_argument("--num_layers", default=config['NUM_LAYERS'], type=int)
        parser.add_argument("--dropout", default=config['DROPOUT'], type=float)
        parser.add_argument("--bridge", default=config['USE_BRIDGE'], type=bool)
        parser.add_argument("--teacher_forcing_ratio", default=config['TEACHER_FORCING_RATIO'], type=float)
        parser.add_argument("--learning_rate", default=config['LEARNING_RATE'], type=float)
        parser.add_argument("--max_len", default=config['MSG_MAX_LEN'] * 2, type=int)
        parser.add_argument("--pad_token_id", default=config['PAD_TOKEN_ID'], type=int)
        return parser
