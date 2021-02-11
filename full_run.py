import torch
import pytorch_lightning as pl

from transformers import RobertaModel, RobertaConfig

import numpy as np

from datasets import load_metric
import nltk
nltk.download('wordnet') # used in meteor metric

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

from dataset_utils.cmg_data_module import CMGDataModule
from faiss_knn import FaissKNN


def decode_preds_and_targets(preds, targets, metric, tokenizer):
    decoded_preds = []
    decoded_targets = []

    for pred, trg in zip(preds, targets):
        # we have k preds for each target
        # choose one with the biggest bleu score
        decoded_pred = tokenizer.batch_decode(pred, skip_special_tokens=True,
                                              clean_up_tokenization_spaces=False)
        decoded_target = tokenizer.decode(trg, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        scores = np.array([metric.compute(predictions=[pr.split()],
                                          references=[[decoded_target.split()]])["bleu"] for pr in decoded_pred])

        decoded_preds.append(decoded_pred[np.argmax(scores)])
        decoded_targets.append(decoded_target)
    return decoded_preds, decoded_targets


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    pl.seed_everything(42)

    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    # load metrics
    bleu_decoding = load_metric("bleu")
    bleu = load_metric("bleu")
    rouge = load_metric("rouge")
    meteor = load_metric("meteor")

    # prepare datamodule
    dm = CMGDataModule(**cfg.dataset)
    dm.setup()

    # use wandb logger
    logger = instantiate(cfg.logger)

    # use randomly initialized RoBERTa as encoder
    encoder_config = RobertaConfig()
    encoder_config.num_hidden_layers = cfg.num_layers_encoder
    encoder = RobertaModel(config=encoder_config)
    encoder.eval()
    # resize embeddings to match vocab with new special token
    encoder.resize_token_embeddings(len(dm._tokenizer))

    # use faiss for nearest neighbors search
    knn = FaissKNN(cfg.k, encoder_config.hidden_size)
    # -------------------------
    #         "train"         -
    # -------------------------
    # (for kNN, just store vectors from RoBERTa)

    dm.setup('fit')

    for batch in tqdm(dm.train_dataloader(), total=int(len(dm.train_dataloader()))):
        # use pooler_output as embedding of each sequence
        # (torch.FloatTensor of shape (batch_size, hidden_size))
        with torch.no_grad():
            embeddings = encoder(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 token_type_ids=batch[2])[1].detach().numpy()
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
            embeddings = encoder(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 token_type_ids=batch[2])[1].detach().numpy()

        test_preds = knn.predict(np.ascontiguousarray(embeddings))

        # decode generated sequences and targets into strings
        preds, targets = decode_preds_and_targets(test_preds, batch[3].detach().numpy(), bleu_decoding, dm._tokenizer)

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
