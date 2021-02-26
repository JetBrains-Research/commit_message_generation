import pytorch_lightning as pl

import torch

from transformers import RobertaForMaskedLM, RobertaTokenizer, AdamW, get_linear_schedule_with_warmup

import wandb

from datasets import load_metric

import nltk

nltk.download('wordnet')


def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False):
    """ Generate a word from from out[gen_idx]
    Copied from

    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k
    """
    logits = out[:, gen_idx]

    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)

    return idx


class RobertaForMLMModule(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 encoder_name_or_path: str,
                 tokenizer: RobertaTokenizer,
                 num_epochs: int,
                 num_batches: int,
                 **kwargs):
        super().__init__()

        self._tokenizer = tokenizer
        self._num_epochs = num_epochs
        self._num_batches = num_batches
        self.learning_rate = learning_rate

        self.save_hyperparameters()

        # use CodeBERT
        self.model = RobertaForMaskedLM.from_pretrained(encoder_name_or_path)

        print("\n====MODEL CONFIG====\n")
        print(self.model.config)
        print()

        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")
        self.meteor = load_metric("meteor")

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def forward(self, batch):
        self.examples_count += len(batch['input_ids'])
        return self.model(input_ids=batch['input_ids'],
                          attention_mask=batch['attention_mask'],
                          labels=batch['labels'])

    def generate(self, batch):
        """ Generate one word at a time, in L->R order"""
        max_len = batch['input_ids'].shape[1]
        top_k = 0
        temperature = None
        sample = True

        outputs = batch['input_ids'].detach().clone()

        for i, input in enumerate(outputs):
            eos_ids = torch.where(input == self._tokenizer.eos_token_id)[0]
            seed_text = input[:(eos_ids[0] + 2)].tolist()  # diff is our seed_text, it ends with first [EOS] token
            seed_len = len(seed_text)

            outputs[i] = torch.tensor(seed_text + [self._tokenizer.mask_token_id] * (max_len - seed_len - 1) \
                                      + [self._tokenizer.eos_token_id])

            for iter in range(max_len - seed_len - 1):
                inp = torch.tensor(outputs[i][:seed_len + iter].tolist() + [self._tokenizer.eos_token_id]).to(
                    self.device)
                out = self({'input_ids': inp.view(1, -1),
                            'attention_mask': torch.ones_like(inp.view(1, -1)).to(self.device),
                            'labels': None}).logits
                idxs = generate_step(out, gen_idx=seed_len + iter, top_k=top_k, temperature=temperature, sample=sample)
                outputs[i][seed_len + iter] = idxs
        return outputs

    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)[:2]

        table = None

        self.logger.experiment.log({"train_loss_step": loss}, step=self.examples_count)
        return {"loss": loss, "examples": table}

    def training_epoch_end(self, outputs):
        train_loss_mean = torch.stack([x["loss"] for x in outputs]).mean()
        tables = [x["examples"] for x in outputs if x["examples"] is not None]
        try:
            self.logger.experiment.log({"train_examples": tables[0],
                                        "train_loss_epoch": train_loss_mean}, step=self.examples_count)
        except IndexError:
            self.logger.experiment.log({"train_loss_epoch": train_loss_mean}, step=self.examples_count)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        loss, logits = self(batch)[:2]
        # generate
        gen_sequence = self.generate(batch)
        # decode generated sequences and targets into strings
        preds, targets = self.decode_preds_and_targets(gen_sequence, batch['target_input_ids'])
        # create a little table with examples
        table = self.make_wandb_table(preds, targets)
        # add batches to metrics
        self.bleu.add_batch(predictions=[line.split() for line in preds],
                            references=[[line.split()] for line in targets])
        self.rouge.add_batch(predictions=preds, references=targets)
        self.meteor.add_batch(predictions=preds, references=targets)
        return {"val_loss": loss, "examples": table}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()

        bleu = self.bleu.compute()
        rouge = self.rouge.compute()
        meteor = self.meteor.compute()
        self.logger.experiment.log({"val_examples": outputs[0]["examples"],
                                    "val_bleu": bleu["bleu"],
                                    "val_rouge1": rouge["rouge1"].mid.fmeasure,
                                    "val_rouge2": rouge["rouge2"].mid.fmeasure,
                                    "val_rougeL": rouge["rougeL"].mid.fmeasure,
                                    "val_meteor": meteor["meteor"],
                                    "val_loss": val_loss_mean}, step=self.examples_count)

    def test_step(self, batch, batch_idx):
        # generate
        gen_sequence = self.generate(batch)
        # decode generated sequences and targets into strings
        preds, targets = self.decode_preds_and_targets(gen_sequence, batch['target_input_ids'])
        # create a little table with examples
        table = self.make_wandb_table(preds, targets)
        # add batches to metrics
        self.bleu.add_batch(predictions=[line.split() for line in preds],
                            references=[[line.split()] for line in targets])
        self.rouge.add_batch(predictions=preds, references=targets)
        self.meteor.add_batch(predictions=preds, references=targets)
        return {"examples": table}

    def test_epoch_end(self, outputs):
        bleu = self.bleu.compute()
        rouge = self.rouge.compute()
        meteor = self.meteor.compute()
        self.logger.experiment.log({"test_examples": outputs[0]["examples"],
                                    "test_bleu": bleu["bleu"],
                                    "test_rouge1": rouge["rouge1"].mid.fmeasure,
                                    "test_rouge2": rouge["rouge2"].mid.fmeasure,
                                    "test_rougeL": rouge["rougeL"].mid.fmeasure,
                                    "test_meteor": meteor["meteor"]}, step=self.examples_count)

    def decode_preds_and_targets(self, preds, targets):
        dec_targets = self._tokenizer.batch_decode(targets,
                                                   skip_special_tokens=True, clean_up_tokenization_spaces=False)
        dec_preds = [self._tokenizer.decode(pred[torch.where(pred == self._tokenizer.eos_token_id)[0][0] + 1:],
                                            skip_special_tokens=True, clean_up_tokenization_spaces=False)
                     for pred in preds]
        return dec_preds, dec_targets

    def make_wandb_table(self, preds, targets, n_examples=8):
        # create a little wandb table with examples
        table = wandb.Table(columns=["Predicted", "Target"])

        for i in range(n_examples):
            try:
                table.add_data(preds[i],
                               targets[i])
            except IndexError:
                break
        return table

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer, self._num_batches * 5,
                                                                  self._num_epochs * self._num_batches),
                     'interval': 'step',
                     'frequency': 1}
        return [optimizer], [scheduler]
