import pytorch_lightning as pl

import torch
import torch.nn.functional as F

from transformers import EncoderDecoderModel, RobertaModel, RobertaConfig, GPT2LMHeadModel, GPT2Config, \
    RobertaTokenizer, GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

import wandb

from datasets import load_metric

import nltk

nltk.download('wordnet')

from copy import copy


def remove_layers_from(teacher, num_layers, model_type):
    if model_type == 'roberta':
        teacher_config = teacher.config
        student_config = copy(teacher.config)
        student_config.num_hidden_layers = num_layers
        student = RobertaModel(config=student_config)

        # copy all embeddings
        student.embeddings.word_embeddings = teacher.embeddings.word_embeddings
        student.embeddings.position_embeddings = teacher.embeddings.position_embeddings
        student.embeddings.token_type_embeddings = teacher.embeddings.token_type_embeddings
        student.embeddings.LayerNorm = teacher.embeddings.LayerNorm
        student.embeddings.dropout = teacher.embeddings.dropout

        # uniformly pick from middle layers from teacher
        # it is basically np.linspace(0, teacher_config.num_hidden_layers,
        #                             num=student_config.num_hidden_layers, endpoint=True)
        step = (teacher_config.num_hidden_layers - 1) / (student_config.num_hidden_layers - 1)
        for student_layer, teacher_layer in enumerate(int(i * step) for i in range(student_config.num_hidden_layers)):
            student.encoder.layer[student_layer] = teacher.encoder.layer[teacher_layer]

    elif model_type == 'gpt':
        teacher_config = teacher.config
        student_config = copy(teacher.config)
        student_config.n_layer = num_layers

        student = GPT2LMHeadModel(config=student_config)

        # Copying all embeddings
        student.transformer.wte = teacher.transformer.wte
        student.transformer.wpe = teacher.transformer.wpe
        student.transformer.drop = teacher.transformer.drop
        # Maybe there is something else in BERT that need to be copied!
        # Specific thing for GPT2LMHead. Not necessary for BERT
        student.tie_weights()
        # Uniformly pick from middle layers from teacher
        # It is basically np.linspace(0, teacher_config.n_layer, num=student_config.n_layer, endpoint=True)
        step = (teacher_config.n_layer - 1) / (student_config.n_layer - 1)
        for student_layer, teacher_layer in enumerate(int(i * step) for i in range(student_config.n_layer)):
            student.transformer.h[student_layer] = teacher.transformer.h[teacher_layer]

    return student


class EncoderDecoderModule(pl.LightningModule):
    def __init__(self,
                 learning_rate: float,
                 encoder_name_or_path: str,
                 decoder_name_or_path: str,
                 unfreeze_encoder_after: int,
                 freeze_encoder_after: int,
                 num_layers_encoder: int,
                 num_layers_decoder: int,
                 src_tokenizer: RobertaTokenizer,
                 trg_tokenizer: GPT2Tokenizer,
                 num_epochs: int,
                 num_batches: int,
                 **kwargs):
        super().__init__()

        self._unfreeze_after = unfreeze_encoder_after
        self._freeze_after = freeze_encoder_after
        self._src_tokenizer = src_tokenizer
        self._trg_tokenizer = trg_tokenizer
        self._num_epochs = num_epochs
        self._num_batches = num_batches

        self.save_hyperparameters()

        self.learning_rate = learning_rate

        # use CodeBERT as encoder and remove part of the layers
        encoder = remove_layers_from(RobertaModel.from_pretrained(encoder_name_or_path),
                                     num_layers_encoder, 'roberta')

        # resize embeddings to match vocab with new special token
        encoder.resize_token_embeddings(len(self._src_tokenizer))

        # change token_type_embeddings dimension to 2
        encoder.config.type_vocab_size = 2
        encoder.embeddings.token_type_embeddings = torch.nn.Embedding.from_pretrained(
            torch.cat((encoder.embeddings.token_type_embeddings.weight,
                       encoder.embeddings.token_type_embeddings.weight), dim=0))

        # use distilGPT-2 as decoder and remove part of the layer
        decoder_config = GPT2Config.from_pretrained(decoder_name_or_path)
        decoder_config.is_decoder = True
        decoder_config.add_cross_attention = True
        decoder = remove_layers_from(GPT2LMHeadModel.from_pretrained(decoder_name_or_path, config=decoder_config),
                                     num_layers_decoder, 'gpt')

        self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

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

        self.bleu = load_metric("bleu")
        self.rouge = load_metric("rouge")
        self.meteor = load_metric("meteor")

        # to make logs for different batch sizes prettier
        self.examples_count = 0

    # def on_train_epoch_start(self) -> None:
    # unfreeze codebert on certain epoch
    # if self.current_epoch == self._unfreeze_after:
    #    for param in self.model.encoder.parameters():
    #        param.requires_grad = True

    # freeze codebert on certain epoch
    # if self.current_epoch == self._freeze_after:
    #    for param in self.model.encoder.parameters():
    #        param.requires_grad = False

    def forward(self, batch):
        self.examples_count += len(batch[0])

        encoder_outputs = self.model.encoder(
            input_ids=batch[0],
            attention_mask=batch[1],
            token_type_ids=batch[2])

        # transformers assume pad indices to be -100
        # gpt2 has no pad tokens so use attention mask
        return self.model(encoder_outputs=encoder_outputs,
                          decoder_input_ids=batch[3],
                          decoder_attention_mask=batch[4],
                          labels=batch[3].where(batch[4].type(torch.ByteTensor).to(self.device),
                                                torch.tensor(-100, device=self.device)))

    def generate(self, batch):
        return self.model.generate(input_ids=batch[0],
                                   attention_mask=batch[1],
                                   token_type_ids=batch[2])

    def training_step(self, batch, batch_idx):
        loss, logits = self(batch)[:2]

        # log train examples on every 1000th batch in epoch
        if self.global_step % 1000 == 0:
            with torch.no_grad():
                gen_sequence = self.generate(batch)
                preds, targets = self.decode_preds_and_targets(gen_sequence, batch[3])
                table = self.make_wandb_table(batch[0], preds, targets)
        else:
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
        preds, targets = self.decode_preds_and_targets(gen_sequence, batch[3])
        # create a little table with examples
        table = self.make_wandb_table(batch[0], preds, targets)
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
        loss, logits = self(batch)[:2]
        # generate
        gen_sequence = self.generate(batch)
        # decode generated sequences and targets into strings
        preds, targets = self.decode_preds_and_targets(gen_sequence, batch[3])
        # create a little table with examples
        table = self.make_wandb_table(batch[0], preds, targets)
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

    def decode_preds_and_targets(self, generated, target):
        if target.shape[1] > generated.shape[1]:
            # pad generated tokens to match sequence length dimension with target
            generated = F.pad(input=generated, pad=(0, target.shape[1] - generated.shape[1], 0, 0), mode='constant',
                              value=self._trg_tokenizer.pad_token_id)
        elif generated.shape[1] > target.shape[1]:
            # pad target tokens to match sequence length dimension with generated
            target = F.pad(input=target, pad=(0, generated.shape[1] - target.shape[1], 0, 0), mode='constant',
                           value=self._trg_tokenizer.pad_token_id)

        # decoded preds and targets
        targets = self._trg_tokenizer.batch_decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        preds = self._trg_tokenizer.batch_decode(generated, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
        return preds, targets

    def make_wandb_table(self, source, preds, targets, n_examples=8):
        # create a little wandb table with examples
        table = wandb.Table(columns=["Source Before", "Source After", "Predicted", "Target"])

        # find sequences before and after in source
        end = torch.where(source == self._src_tokenizer.eos_token_id)[1][1::3]

        for i in range(n_examples):
            try:
                table.add_data(self._src_tokenizer.decode(source[i, :end[i] + 1], skip_special_tokens=True, \
                                                          clean_up_tokenization_spaces=False),  # decode sequence before
                               self._src_tokenizer.decode(source[i, end[i] + 1:], skip_special_tokens=True, \
                                                          clean_up_tokenization_spaces=False),  # decode sequence after
                               preds[i],
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
