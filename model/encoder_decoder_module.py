from argparse import ArgumentParser

import pytorch_lightning as pl

import torch
import torch.nn.functional as F

from transformers import RobertaConfig, RobertaModel, AdamW

from model.decoder import Decoder

from metrics import AccuracyMetric
from metrics import BleuMetric


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self,
                 embedding_dim,
                 hidden_size_decoder,
                 hidden_size_encoder,
                 num_heads,
                 num_layers,
                 dropout,
                 bridge,
                 teacher_forcing_ratio,
                 learning_rate,
                 model_name_or_path,
                 tokenizer,
                 **kwargs):
        super().__init__()

        self.tokenizer = tokenizer
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id

        self.encoder_config = RobertaConfig.from_pretrained(model_name_or_path)
        self.encoder = RobertaModel.from_pretrained(model_name_or_path, config=self.encoder_config)

        self.decoder = Decoder(embed_dim=embedding_dim,
                               vocab_size=tokenizer.vocab_size,
                               hidden_size=hidden_size_decoder,
                               hidden_size_encoder=hidden_size_encoder,
                               num_heads=num_heads,
                               num_layers=num_layers,
                               dropout=dropout,
                               bridge=bridge,
                               teacher_forcing_ratio=teacher_forcing_ratio)

        self.accuracy = AccuracyMetric(self.pad_token_id)
        self.bleu = BleuMetric()

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

    def test_step(self, batch, batch_idx):
        # TODO: implement beam search and add choice between beam search/greedy approaches
        src, trg = batch
        encoder_output, _, encoder_final = self.encoder(input_ids=src['input_ids'],
                                                        attention_mask=src['attention_mask'],
                                                        output_hidden_states=True)
        t = encoder_final[0].shape[1] - 1
        encoder_final = torch.stack(encoder_final)[:, :, t, :][-self.decoder.num_layers:, :]

        prev_y = torch.ones(src['input_ids'].shape[0], 1).fill_(self.bos_token_id).type_as(src['input_ids'])
        prev_y_mask = torch.ones_like(prev_y)

        preds = torch.zeros((src['input_ids'].shape[0], trg['input_ids'].shape[1]))
        hidden = None

        for i in range(trg['input_ids'].shape[1]):
            decoder_states, hidden, output = self.decoder(prev_y,
                                                          prev_y_mask,
                                                          encoder_output, encoder_final,
                                                          torch.logical_not(src['attention_mask']), hidden=hidden)
            _, next_word = torch.max(output, dim=2)
            preds[:, i] = torch.flatten(next_word)
            prev_y = next_word  # change prev id to generated id
        preds = preds.detach().cpu()
        return {'preds': preds, 'targets': trg['input_ids']}

    def test_epoch_end(self, outputs):
        # TODO: should we compute accuracy on decoded strings or with token ids (probably faster)?
        # compute accuracy with tensors
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu()
        targets = torch.cat([x['targets'] for x in outputs]).detach().cpu()

        acc = self.accuracy(preds, targets)

        # compute BLEU with decoded strings
        targets = [
            self.tokenizer.decode(example, skip_special_tokens=True, clean_up_tokenization_spaces=False).split(' ')
            for example in targets.tolist()]

        preds = [self.tokenizer.decode(example, skip_special_tokens=True, clean_up_tokenization_spaces=False).split(' ')
                 for example in preds.tolist()]

        bleu = self.bleu(preds, targets)

        self.log('test_accuracy', acc, prog_bar=True, logger=True)
        self.log('test_bleu', bleu, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
