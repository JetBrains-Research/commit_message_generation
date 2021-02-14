import torch
import pytorch_lightning as pl

from transformers import RobertaModel, RobertaConfig

import numpy as np

from datasets import load_metric
import nltk

nltk.download('wordnet')  # used in meteor metric

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

from dataset_utils.cmg_data_module import CMGDataModule
from faiss_knn import FaissKNN


def decode_preds_and_targets(preds, targets, tokenizer):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)

    decoded_targets = tokenizer.batch_decode(targets, skip_special_tokens=True,
                                             clean_up_tokenization_spaces=False)

    return decoded_preds, decoded_targets


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    pl.seed_everything(42)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    # load metrics
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    meteor = load_metric("meteor")

    # prepare datamodule
    dm = CMGDataModule(**cfg.dataset)
    dm.setup('fit')

    # use wandb logger
    logger = instantiate(cfg.logger)

    # use CodeBERT as encoder
    encoder = RobertaModel.from_pretrained('microsoft/codebert-base')

    # resize embeddings to match vocab with new special token
    encoder.resize_token_embeddings(len(dm._tokenizer))

    encoder.to("cuda")

    print("Model\n", encoder)

    # use faiss for nearest neighbors search
    knn = FaissKNN(1, encoder.config.hidden_size)
    # -------------------------
    #         "train"         -
    # -------------------------
    # (for kNN, just store vectors from RoBERTa)

    for batch in tqdm(dm.train_dataloader(), total=int(len(dm.train_dataloader()))):
        # use pooler_output as embedding of each sequence
        # (torch.FloatTensor of shape (batch_size, hidden_size))
        with torch.no_grad():
            embeddings = encoder(input_ids=batch[0].to("cuda"),
                                 attention_mask=batch[1].to("cuda"))[1].cpu().detach().numpy()
        knn.fit(np.ascontiguousarray(embeddings), batch[3].detach().numpy())

    # ----------------------
    #         test         -
    # ----------------------
    # lazy load test data
    dm.setup('test')

    for batch in tqdm(dm.test_dataloader(), total=int(len(dm.test_dataloader()))):
        # use pooler_output as embedding of each sequence
        # (torch.FloatTensor of shape (batch_size, hidden_size))
        with torch.no_grad():
            embeddings = encoder(input_ids=batch[0].to("cuda"),
                                 attention_mask=batch[1].to("cuda"))[1].cpu().detach().numpy()

        test_preds = knn.predict(np.ascontiguousarray(embeddings))

        # decode generated sequences and targets into strings
        preds, targets = decode_preds_and_targets(np.squeeze(test_preds, axis=1), batch[3].detach().numpy(),
                                                  dm._tokenizer)

        # add batches to metrics
        bleu.add_batch(predictions=[line.split() for line in preds],
                       references=[[line.split()] for line in targets])
        rouge.add_batch(predictions=preds, references=targets)
        meteor.add_batch(predictions=preds, references=targets)

    bleu_res = bleu.compute()
    rouge_res = rouge.compute()
    meteor_res = meteor.compute()

    logger.experiment.log({"test_bleu": bleu_res["bleu"],
                           "test_rouge1": rouge_res["rouge1"].mid.fmeasure,
                           "test_rouge2": rouge_res["rouge2"].mid.fmeasure,
                           "test_rougeL": rouge_res["rougeL"].mid.fmeasure,
                           "test_meteor": meteor_res["meteor"]})


if __name__ == '__main__':
    main()
