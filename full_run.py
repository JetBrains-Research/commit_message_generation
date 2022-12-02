import json
import logging

import hydra
import jsonlines
import pandas as pd
import pytorch_lightning as pl
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data_utils.cmg_data_module import CMGDataModule
from src.embedders import BagOfWordsEmbedder, TransformerEmbedder
from src.search import DiffSearch


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # -----------------------
    #          init         -
    # -----------------------
    cfg.dataset.dataset_root = to_absolute_path(cfg.dataset.dataset_root)

    pl.seed_everything(42)

    # init logger (Weights & Biases)
    run = None
    if cfg.logger:
        run = wandb.init(
            **cfg.logger.args,
            tags=[
                cfg.configuration,
            ],
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
        )

    # init datamodule
    dm = CMGDataModule(**cfg.dataset)

    # init embedder
    embedder_conf = cfg.embedder[cfg.configuration]
    if cfg.configuration == "bag_of_words":
        logging.info("Using Bag-of-Words embedder")
        vocabulary = None
        if embedder_conf.load_vocab:
            logging.info("Loading pretrained vocabulary...")
            with open(embedder_conf.vocab_filename, "r") as f:
                vocabulary = json.load(f)
        embedder = BagOfWordsEmbedder(vocabulary=vocabulary, max_features=embedder_conf.max_features)
        logging.info("Building vocabulary...")
        embedder.build_vocab(input_filename=dm.train_path, chunksize=embedder_conf.chunksize)
        if embedder_conf.save_vocab:
            embedder.save_vocab(vocab_filename=f"vocab_{embedder_conf.max_features}.json")
    elif cfg.configuration == "transformer":
        logging.info("Using Transformer embedder")
        embedder = TransformerEmbedder(**embedder_conf)
    else:
        raise ValueError(f"Unknown embedder '{cfg.embedder.configuration}'")

    # setup datasets
    dm.setup("fit")

    # init retrieval class
    search_conf = cfg.search[cfg.configuration]
    if search_conf.configuration == "diff":
        search_conf = search_conf["diff"]
        logging.info("Using diff-based retrieval")
        search = DiffSearch(
            num_neighbors=search_conf.num_neighbors,
            num_trees=search_conf.num_trees,
            embeddings_dim=embedder.embeddings_dim,
            input_fname=dm.train_path,
        )
    else:
        raise ValueError(f"Unknown search '{search_conf.configuration}'")

    # -------------------------------
    #              train            -
    #     (build embeddings index)  -
    # -------------------------------
    if search_conf.load_index:
        logging.info("Loading pretrained index")
        if "index_fname" not in search_conf:
            raise ValueError("Please, pass path to pretrained index to config.")
        search.load_index(search_conf.index_fname)
    else:
        for batch in tqdm(dm.train_dataloader(), desc="Building embeddings index"):
            search.add_batch(embedder.transform(batch))
        search.finalize()

    # ----------------------
    #         test         -
    # ----------------------
    dm.setup("test")

    test_predictions = []
    test_inputs = []
    for batch in tqdm(dm.test_dataloader(), desc="Retrieving test predictions"):
        test_predictions.extend(search.predict_batch(embedder.transform(batch)))
        test_inputs.extend(batch)

    logging.info("Saving predictions")
    with jsonlines.open("test_predictions.jsonl", "w") as writer:
        writer.write_all(test_predictions)

    if run and cfg.logger.log_artifact:
        logging.info("Adding artifact to W&B")
        wandb.Table.MAX_ROWS = 50000
        artifact = wandb.Artifact(cfg.logger.artifact.name, type=cfg.logger.artifact.type)
        preds_df = pd.DataFrame(test_predictions).add_suffix("_prediction")
        inputs_ds = pd.DataFrame(test_inputs).add_suffix("_input")
        df = pd.concat([inputs_ds, preds_df], axis=1).rename(
            columns={
                "idx_input": "Input id",
                "diff_input": "Input diff",
                "message_input": "Input message",
                "idx_prediction": "Retrieved Id",
                "diff_prediction": "Retrieved diff",
                "message_prediction": "Retrieved message",
                "distance_prediction": "Distance",
            }
        )
        artifact.add(
            wandb.Table(dataframe=df[["Id", "Input diff", "Retrieved diff", "Prediction", "Target", "Distance"]]),
            cfg.logger.artifact.table_name,
        )
        run.log_artifact(artifact, aliases=[f"{cfg.configuration}_{search_conf.num_neighbors}NN"])


if __name__ == "__main__":
    main()
