import pytorch_lightning as pl

import numpy as np

import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer

import wandb

from metrics import accuracy_MRR


class GPT2LMHeadModule(pl.LightningModule):
    def __init__(self,
                 decoder_name_or_path: str,
                 tokenizer: GPT2Tokenizer,
                 **kwargs):
        super().__init__()

        self._tokenizer = tokenizer
        self.save_hyperparameters()

        # use pretrained GPT-2 as decoder
        self.model = GPT2LMHeadModel.from_pretrained(decoder_name_or_path)

        print("\n====MODEL CONFIG====\n")
        print(self.model.config)
        print()

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def forward(self, batch):
        self.examples_count += len(batch[0])

        return self.model(input_ids=batch[0],
                          attention_mask=batch[1],
                          labels=batch[2])

    def test_step(self, batch, batch_idx):
        scores = self(batch).logits

        # calculate metrics
        acc_top1, acc_top5, MRR_top5 = accuracy_MRR(scores, batch[2], top_k=5, shift=True)

        # get top k predictions for each token in each batch
        _, top_k_predictions = torch.topk(scores, k=5)

        # assign target pad tokens idx to pad_token_id to avoid logging them in table
        top_k_predictions[batch[2] == -100, :] = self._tokenizer.pad_token_id

        targets_no_history = batch[2]
        targets_no_history[targets_no_history == -100] = self._tokenizer.pad_token_id

        # decode top k predictions and targets
        decoded_preds, decoded_targets_no_history, decoded_targets_w_history = \
            self.decode_preds_and_targets(top_k_predictions,
                                          batch[0],
                                          targets_no_history)

        # log a little table with examples
        table = self.make_wandb_table(decoded_preds, decoded_targets_no_history, decoded_targets_w_history)
        return {"acc_top1": acc_top1, "acc_top5": acc_top5, "MRR_top5": MRR_top5, "examples": table}

    def test_epoch_end(self, outputs):
        test_acc_top1 = np.mean([x["acc_top1"] for x in outputs])
        test_acc_top5 = np.mean([x["acc_top5"] for x in outputs])
        test_MRR_top5 = np.mean([x["MRR_top5"] for x in outputs])
        self.logger.experiment.log({"test_examples": outputs[0]["examples"],
                                    "test_acc_top1": test_acc_top1,
                                    "test_acc_top5": test_acc_top5,
                                    "test_MRR_top5": test_MRR_top5},
                                   step=self.examples_count)

    def decode_preds_and_targets(self, preds, targets_w_history, targets_no_history):
        # decoded preds and targets
        decoded_targets_w_history = self._tokenizer.batch_decode(targets_w_history, skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=False)
        decoded_targets_no_history = self._tokenizer.batch_decode(targets_no_history, skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=False)
        decoded_preds = [self._tokenizer.batch_decode(preds[:, :, i], skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=False) for i in range(5)]

        return decoded_preds, decoded_targets_no_history, decoded_targets_w_history

    def make_wandb_table(self, decoded_preds, decoded_targets_no_history, decoded_targets_w_history, n_examples=8):
        # create a little wandb table with examples
        cols = ["Target (with history)", "Target"]
        cols.extend([f'Top {i + 1}' for i in range(5)])
        table = wandb.Table(columns=cols)

        for i in range(n_examples):
            try:
                table.add_data(decoded_targets_w_history[i],
                               decoded_targets_no_history[i],
                               decoded_preds[0][i],
                               decoded_preds[1][i],
                               decoded_preds[2][i],
                               decoded_preds[3][i],
                               decoded_preds[4][i])
            except IndexError:
                break

        return table
