import logging
import os
from typing import List, Optional

import hydra
import jsonlines
import wandb
from tqdm import tqdm

from conf import RetrievalConfig
from src.data_utils import CMCDataModule
from src.model import CMCModule
from src.retrieval import DiffSearch, TransformerEmbedder
from src.retrieval.utils import RetrievalPrediction
from src.utils import WandbOrganizer


def download_artifact(cfg: RetrievalConfig, run: wandb.wandb_sdk.wandb_run.Run) -> str:
    """Helper function to download relevant artifact from W&B.

    Args:
        cfg: Current configuration, necessary to find relevant artifact.
        run: Current W&B run.

    Returns:
        A local path to the artifact.
    """
    artifact_name = f"{cfg.logger.input_artifact.project}/{run.name}:{cfg.logger.input_artifact.version}"
    artifact = run.use_artifact(artifact_name)
    if "tags" in artifact.metadata:
        run.tags = artifact.metadata["tags"]

    artifact.get_path(cfg.logger.input_artifact.artifact_path).download(
        root=hydra.utils.to_absolute_path(f"{cfg.logger.input_artifact.local_path}/{run.name}")
    )

    return os.path.join(
        hydra.utils.to_absolute_path(f"{cfg.logger.input_artifact.local_path}/{run.name}"),
        cfg.logger.input_artifact.artifact_path,
    )


def export_model_checkpoint(cfg: RetrievalConfig) -> None:
    """Helper function to export model weights in Transformers format from Lightning checkpoint."""
    if cfg.ckpt_path:
        print(f"Checkpoint path: {cfg.ckpt_path}")

        module = CMCModule.load_from_checkpoint(
            cfg.ckpt_path,
            model_cfg=cfg.model,
        )

        os.makedirs(os.path.join(cfg.ckpt_path, "transformers_format"), exist_ok=True)
        module.model.save_pretrained(os.path.join(cfg.ckpt_path, "transformers_format"))


@hydra.main(version_base="1.1", config_path="conf", config_name="retrieve_config")
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
            project=cfg.logger.project, name=f"{run_name}_retrieval", job_type="retrieval"
        )
        assert run is not None

        if cfg.logger.download_artifact:
            logging.info("Downloading artifact from W&B")
            cfg.ckpt_path = download_artifact(run=run, cfg=cfg)
    else:
        run = None

    # ------------------------------
    # -    extract model weights   -
    # ------------------------------
    assert cfg.ckpt_path
    cfg.ckpt_path = hydra.utils.to_absolute_path(cfg.ckpt_path)
    export_model_checkpoint(cfg)

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
    embedder = TransformerEmbedder(name_or_path=os.path.join(cfg.ckpt_path, "transformers_format"), device=cfg.device)

    os.makedirs(hydra.utils.to_absolute_path(cfg.search.index_root_dir), exist_ok=True)
    search = DiffSearch(
        num_trees=cfg.search.num_trees,
        embeddings_dim=embedder.embeddings_dim,
        load_index=cfg.search.load_index,
        index_root_dir=hydra.utils.to_absolute_path(cfg.search.index_root_dir),
    )
    if not cfg.search.load_index:
        for batch in tqdm(dm.retrieval_dataloader(part="train"), desc="Building embeddings index"):
            search.add_batch(embedder.transform(batch))
        search.finalize()

    # ------------------------------
    # -       retrieve NNs         -
    # ------------------------------
    for part in ["train", "val", "test"]:
        logging.info(f"Start processing {part}")

        open(f"{part}_predictions.jsonl", "w").close()
        predictions: List[RetrievalPrediction] = []
        for batch in tqdm(dm.retrieval_dataloader(part=part), desc=f"Retrieving predictions for {part}"):
            if len(predictions) > 1000:
                with jsonlines.open(f"{part}_predictions.jsonl", "a") as writer:
                    writer.write_all(
                        [{"pos_in_file": pred["pos_in_file"], "distance": pred["distance"]} for pred in predictions]
                    )
                predictions = []

            predictions.extend(search.predict_batch(embedder.transform(batch), is_train=(part == "train")))

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
