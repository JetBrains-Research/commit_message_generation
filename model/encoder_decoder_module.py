from argparse import ArgumentParser

import pytorch_lightning as pl

import torch
from torch import nn

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AdamW

import wandb

from model.decoder import Decoder

from metrics import AccuracyMetric
from metrics import BleuMetric


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self,
                 embedding_dim: int,
                 hidden_size_decoder: int,
                 hidden_size_encoder: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float,
                 bridge: bool,
                 teacher_forcing_ratio: float,
                 learning_rate: float,
                 reduction: str,
                 model_name_or_path: str,
                 tokenizer: RobertaTokenizer,
                 **kwargs):
        super().__init__()

        self._tokenizer = tokenizer
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

        self.loss = nn.NLLLoss(reduction=reduction, ignore_index=self.pad_token_id)

        self.accuracy = AccuracyMetric(self.pad_token_id)
        self.bleu = BleuMetric()

    def forward(self, batch):
        src, trg = batch
        # encode step
        encoder_output, encoder_final, _ = self.encoder(input_ids=src['input_ids'],
                                                        attention_mask=src['attention_mask'],
                                                        output_hidden_states=True)
        # decode step
        return self.decoder(trg['input_ids'],
                            trg['attention_mask'],
                            encoder_output, encoder_final.unsqueeze(0),
                            torch.logical_not(src['attention_mask']))

    def training_step(self, batch, batch_idx):
        src, trg = batch
        hidden, output = self(batch)
        train_loss = self.loss(output.view(-1, output.size(-1)), trg['input_ids'].view(-1))
        self.logger.experiment.log({"train_loss_step": train_loss})
        return train_loss

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x['loss'] for x in outputs]).mean().item()
        self.logger.experiment.log({"train_loss_epoch": train_loss_mean})

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        src, trg = batch
        hidden, output = self(batch)
        val_loss = self.loss(output.view(-1, output.size(-1)), trg['input_ids'].view(-1))
        acc, bleu, table = self.greedy_decode(batch)
        return {"val_loss": val_loss, "val_accuracy": acc, "val_bleu": bleu, "examples": table}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean().item()
        val_acc_mean = torch.stack([x["val_accuracy"] for x in outputs]).mean().item()
        val_bleu_mean = torch.stack([x["val_bleu"] for x in outputs]).mean().item()
        self.logger.experiment.log({"val_examples": outputs[0]["examples"],
                                    "val_accuracy": val_acc_mean,
                                    "val_bleu": val_bleu_mean,
                                    "val_loss": val_loss_mean})

    def test_step(self, batch, batch_idx):
        acc, bleu, table = self.greedy_decode(batch)
        return {"test_accuracy": acc, "test_bleu": bleu, "examples": table}

    def test_epoch_end(self, outputs):
        test_acc_mean = torch.stack([x["test_accuracy"] for x in outputs]).mean().item()
        test_bleu_mean = torch.stack([x["test_bleu"] for x in outputs]).mean().item()
        self.logger.experiment.log({"test_examples": outputs[0]["examples"],
                                    "test_accuracy": test_acc_mean,
                                    "test_bleu": test_bleu_mean})

    def greedy_decode(self, batch):
        # TODO: implement beam search and add choice between beam search/greedy approaches
        src, trg = batch
        encoder_output, encoder_final, _ = self.encoder(input_ids=src['input_ids'],
                                                        attention_mask=src['attention_mask'],
                                                        output_hidden_states=True)

        encoder_final = encoder_final.unsqueeze(0)

        prev_y = torch.ones(src['input_ids'].shape[0], 1).fill_(self.bos_token_id).type_as(src['input_ids'])
        prev_y_mask = torch.ones_like(prev_y)
        preds = torch.zeros((src['input_ids'].shape[0], trg['input_ids'].shape[1])).type_as(src['input_ids'])
        hidden = None

        for i in range(trg['input_ids'].shape[1]):
            hidden, output = self.decoder(prev_y,
                                          prev_y_mask,
                                          encoder_output, encoder_final,
                                          torch.logical_not(src['attention_mask']), hidden=hidden)
            _, next_word = torch.max(output, dim=2)
            preds[:, i] = torch.flatten(next_word)
            prev_y = next_word
        # compute accuracy with tensors
        acc = self.accuracy(preds, trg['input_ids'])

        # compute BLEU with decoded strings
        targets = [
            self._tokenizer.decode(example, skip_special_tokens=True, clean_up_tokenization_spaces=False).split(' ')
            for example in trg['input_ids'].tolist()]

        preds = [self._tokenizer.decode(example, skip_special_tokens=True, clean_up_tokenization_spaces=False).split(' ')
                 for example in preds.tolist()]
        bleu = self.bleu(preds, targets)

        # log a little table with examples
        table = wandb.Table(columns=["Source", "Predicted", "Target"])
        srcs = [self._tokenizer.decode(example, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for example in src['input_ids'].tolist()]
        for i in range(5):
            try:
                table.add_data(srcs[i], ' '.join(preds[i]), ' '.join(targets[i]))
            except IndexError:
                break
        return acc, bleu, table

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)
