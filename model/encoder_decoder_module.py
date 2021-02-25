import pytorch_lightning as pl

import numpy as np

import torch

from transformers import EncoderDecoderModel, RobertaModel, RobertaConfig, GPT2LMHeadModel, GPT2Config, \
    RobertaTokenizer, GPT2Tokenizer

import wandb

from metrics import accuracy_MRR


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self,
                 num_layers_encoder: int,
                 num_layers_decoder: int,
                 src_tokenizer: RobertaTokenizer,
                 trg_tokenizer: GPT2Tokenizer,
                 **kwargs):
        super().__init__()

        self.num_layers_encoder = num_layers_encoder
        self.num_layers_decoder = num_layers_decoder
        self._src_tokenizer = src_tokenizer
        self._trg_tokenizer = trg_tokenizer
        self.save_hyperparameters()

        # use randomly initialized RoBERTa as encoder
        encoder_config = RobertaConfig()
        encoder_config.num_hidden_layers = self.num_layers_encoder
        encoder = RobertaModel(config=encoder_config)
        encoder.resize_token_embeddings(len(self._src_tokenizer))

        # use randomly initialized GPT-2 as decoder
        decoder_config = GPT2Config()
        decoder_config.n_layer = self.num_layers_decoder
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        gpt = GPT2LMHeadModel(config=decoder_config)

        self.model = EncoderDecoderModel(encoder=encoder, decoder=gpt)

        # cache is currently not supported by EncoderDecoder framework
        self.model.decoder.config.use_cache = False

        # do not tie output embeddings to input embeddings
        self.model.config.tie_word_embeddings = False

        # set decoding params
        self.model.config.decoder_start_token_id = self._trg_tokenizer.bos_token_id
        self.model.config.bos_token_id = self._trg_tokenizer.bos_token_id
        self.model.config.eos_token_id = self._trg_tokenizer.eos_token_id
        self.model.config.pad_token_id = self._trg_tokenizer.pad_token_id
        self.model.config.max_length = 30
        self.model.config.min_length = 2
        self.model.config.no_repeat_ngram_size = 4
        self.model.config.early_stopping = True
        self.model.config.num_beams = 4

        print("\n====MODEL CONFIG====\n")
        print(self.model.config)
        print()

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    def forward(self, batch):
        self.examples_count += len(batch[0])

        # transformers assume pad indices to be -100
        # gpt2 has no pad tokens so use attention mask
        return self.model(input_ids=batch[0],
                          attention_mask=batch[1],
                          decoder_input_ids=batch[2],
                          decoder_attention_mask=batch[3],
                          labels=batch[2].where(batch[3].type(torch.ByteTensor).to(self.device),
                                                torch.tensor(-100, device=self.device)))

    def test_step(self, batch, batch_idx):
        scores = self(batch).logits
        labels = batch[2].where(batch[3].type(torch.ByteTensor).to(self.device),
                                torch.tensor(-100, device=self.device))

        # calculate metrics
        acc_top1, acc_top5, MRR_top5 = accuracy_MRR(scores, labels, top_k=5, shift=True)

        # get top k predictions for each token in each batch
        _, top_k_predictions = torch.topk(scores, k=5)

        # assign target pad tokens idx to pad_token_id to avoid logging them in table
        top_k_predictions[labels == -100, :] = self._trg_tokenizer.pad_token_id

        # decode top k predictions and targets
        preds, targets = self.decode_preds_and_targets(top_k_predictions, batch[2])

        # log a little table with examples
        table = self.make_wandb_table(batch[0], preds, targets)
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

    def decode_preds_and_targets(self, generated, target):
        # decoded preds and targets
        targets = self._trg_tokenizer.batch_decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        preds = [self._trg_tokenizer.batch_decode(generated[:, :, i], skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False) for i in range(5)]

        return preds, targets

    def make_wandb_table(self, source, preds, targets, n_examples=8):
        # create a little wandb table with examples
        cols = ["Source", "Target"]
        cols.extend([f'Top {i + 1}' for i in range(5)])
        table = wandb.Table(columns=cols)
        decoded_source = self._src_tokenizer.batch_decode(source, skip_special_tokens=True, \
                                                          clean_up_tokenization_spaces=False)
        for i in range(n_examples):
            try:
                table.add_data(decoded_source[i],
                               targets[i],
                               preds[0][i],
                               preds[1][i],
                               preds[2][i],
                               preds[3][i],
                               preds[4][i])
            except IndexError:
                break

        return table
