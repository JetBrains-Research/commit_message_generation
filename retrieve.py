import logging
import os
from typing import List, Optional

import hydra
import jsonlines
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from conf import RetrievalConfig
from src.data_utils import CMCDataModule
from src.model import CMCModule
from src.retrieval import DiffSearch, TransformerEmbedder
from src.retrieval.utils import CommitEmbeddingExample, RetrievalPrediction
from src.utils import WandbOrganizer


def download_artifact(cfg: RetrievalConfig, run: wandb.wandb_sdk.wandb_run.Run, artifact_name: str) -> str:
    """Helper function to download relevant artifact from W&B.

    Args:
        cfg: Current configuration, necessary to find relevant artifact.
        run: Current W&B run.

    Returns:
        A local path to the artifact.
    """
    full_artifact_name = f"{cfg.logger.input_artifact.project}/{artifact_name}:{cfg.logger.input_artifact.version}"
    artifact = run.use_artifact(full_artifact_name)
    if "tags" in artifact.metadata:
        run.tags = artifact.metadata["tags"]

    artifact.get_path(cfg.logger.input_artifact.artifact_path).download(
        root=hydra.utils.to_absolute_path(f"{cfg.logger.input_artifact.local_path}/{artifact_name}")
    )

    return os.path.join(
        hydra.utils.to_absolute_path(f"{cfg.logger.input_artifact.local_path}/{artifact_name}"),
        cfg.logger.input_artifact.artifact_path,
    )


def export_model_checkpoint(cfg: RetrievalConfig) -> str:
    """Helper function to export model weights in Transformers format from Lightning checkpoint."""
    logging.info(f"Checkpoint path: {cfg.ckpt_path}")

    module = CMCModule.load_from_checkpoint(
        cfg.ckpt_path,
        model_cfg=cfg.model,
    )

    transformers_ckpt_path = os.path.join(cfg.ckpt_path.split("/")[-1], "transformers_format")
    os.makedirs(transformers_ckpt_path, exist_ok=True)
    module.save_pretrained(transformers_ckpt_path)
    return transformers_ckpt_path


@hydra.main(version_base="1.1", config_path="conf", config_name="retrieval_config")
def main(cfg: RetrievalConfig) -> None:
    run_name = WandbOrganizer.get_run_name(
        cfg.model,
        encoder_input_type=cfg.input.encoder_input_type,
        train_with_history=cfg.input.train_with_history,
    )

    # --------------------
    # -     init W&B     -
    # --------------------
    run: Optional[wandb.wandb_sdk.wandb_run.Run]
    if cfg.logger.use_wandb:
        if cfg.logger.use_api_key:
            with open(hydra.utils.to_absolute_path("wandb_api_key.txt"), "r") as f:
                os.environ["WANDB_API_KEY"] = f.read().strip()

        run = wandb.init(  # type: ignore[assignment]
            project=cfg.logger.project,
            name=f"{run_name}_retrieval",
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
            job_type="retrieval",
        )
        assert run is not None

        if cfg.logger.download_artifact:
            logging.info("Downloading artifact from W&B")
            cfg.ckpt_path = download_artifact(run=run, cfg=cfg, artifact_name=run_name)
    else:
        run = None

    # ------------------------------
    # -    extract model weights   -
    # ------------------------------
    assert cfg.ckpt_path
    cfg.ckpt_path = hydra.utils.to_absolute_path(cfg.ckpt_path)
    transformers_ckpt_path = export_model_checkpoint(cfg)

    # ----------------------------
    # -    preprocess data      -
    # ----------------------------
    dm = CMCDataModule(
        dataset_cfg=cfg.dataset,
        model_cfg=cfg.model,
        input_cfg=cfg.input,
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
        world_size=1,
        shift_labels=False,
        process_retrieved=False,
    )
    dm.prepare_data(stage="retrieve")
    dm.setup()

    # -----------------------------
    # -   build embeddings index  -
    # -----------------------------
    embedder = TransformerEmbedder(
        name_or_path=transformers_ckpt_path,
        device=cfg.embedder.device,
        precision=cfg.embedder.precision,
        normalize_embeddings=cfg.embedder.normalize_embeddings,
    )

    os.makedirs(hydra.utils.to_absolute_path(cfg.search.index_root_dir), exist_ok=True)
    search = DiffSearch(
        num_trees=cfg.search.num_trees,
        embeddings_dim=embedder.embeddings_dim,
        load_index=cfg.search.load_index,
        index_root_dir=hydra.utils.to_absolute_path(cfg.search.index_root_dir),
        load_index_path=hydra.utils.to_absolute_path(cfg.search.load_index_path),
    )

    if not cfg.search.load_index:
        for batch in tqdm(dm.retrieval_dataloader(part="train"), desc="Building embeddings index"):
            search.add_batch(embedder.transform(batch))
        search.finalize()

    # ------------------------------
    # -       retrieve NNs         -
    # ------------------------------

    logging.info(f"Start processing train")

    open(f"train_predictions.jsonl", "w").close()
    predictions: List[RetrievalPrediction] = []
    for batch in tqdm(dm.retrieval_dataloader(part="train"), desc="Retrieving predictions for train"):
        if len(predictions) > 10000:
            with jsonlines.open("train_predictions.jsonl", "a") as writer:
                writer.write_all(
                    [{"pos_in_file": pred["pos_in_file"], "distance": pred["distance"]} for pred in predictions]
                )
            predictions = []

        predictions.extend(search.predict_batch_train([example.pos_in_file for example in batch]))

    if len(predictions) > 0:
        with jsonlines.open("train_predictions.jsonl", "a") as writer:
            writer.write_all(
                [{"pos_in_file": pred["pos_in_file"], "distance": pred["distance"]} for pred in predictions]
            )

    logging.info(f"Finish processing train")

    for part in ["val", "test"]:
        logging.info(f"Start processing {part}")

        open(f"{part}_predictions.jsonl", "w").close()
        predictions: List[RetrievalPrediction] = []  # type: ignore[no-redef]
        for batch in tqdm(dm.retrieval_dataloader(part=part), desc=f"Retrieving predictions for {part}"):
            if len(predictions) > 10000:
                with jsonlines.open(f"{part}_predictions.jsonl", "a") as writer:
                    writer.write_all(
                        [{"pos_in_file": pred["pos_in_file"], "distance": pred["distance"]} for pred in predictions]
                    )
                predictions = []

            predictions.extend(search.predict_batch(embedder.transform(batch)))

        if len(predictions) > 0:
            with jsonlines.open(f"{part}_predictions.jsonl", "a") as writer:
                writer.write_all(
                    [{"pos_in_file": pred["pos_in_file"], "distance": pred["distance"]} for pred in predictions]
                )

        logging.info(f"Finish processing {part}")

    # -------------------
    # - log predictions -
    # -------------------
    if run and cfg.logger.upload_artifact:
        logging.info("Uploading artifact to W&B")
        artifact = wandb.Artifact(f"{run_name}_retrieval", type="retrieval")
        artifact.add_file("train_predictions.jsonl")
        artifact.add_file("val_predictions.jsonl")
        artifact.add_file("test_predictions.jsonl")
        run.log_artifact(artifact)


if __name__ == "__main__":
    main()
