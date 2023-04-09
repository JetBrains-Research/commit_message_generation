import logging
import os

import hydra
import jsonlines
import pandas as pd
import wandb
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

from conf import MetricsConfig
from src.utils import EvaluationMetrics

logger = logging.getLogger("datasets")
logger.setLevel(logging.ERROR)


def load_predictions(run: wandb.wandb_sdk.wandb_run.Run, cfg: MetricsConfig) -> str:
    input_artifact = run.use_artifact(
        f"{cfg.logger.artifact_config.project}/{cfg.logger.artifact_config.name}:{cfg.logger.artifact_config.version}"
    )
    if "tags" in input_artifact.metadata:
        run.tags = input_artifact.metadata["tags"]

    input_artifact.get_path(cfg.logger.artifact_config.artifact_path).download(
        root=hydra.utils.to_absolute_path(
            f"{cfg.logger.artifact_config.local_path}/{cfg.logger.artifact_config.name}/predictions"
        )
    )

    predictions_path = os.path.join(
        hydra.utils.to_absolute_path(
            f"{cfg.logger.artifact_config.local_path}/{cfg.logger.artifact_config.name}/predictions"
        ),
        cfg.logger.artifact_config.artifact_path,
    )
    return predictions_path


@hydra.main(version_base="1.1", config_path="conf", config_name="metrics_config")
def main(cfg: MetricsConfig):
    # -----------------------
    #          init         -
    # -----------------------
    if cfg.logger.use_wandb:
        if cfg.logger.use_api_key:
            with open(hydra.utils.to_absolute_path("wandb_api_key.txt"), "r") as f:
                os.environ["WANDB_API_KEY"] = f.read().strip()

        run: wandb.wandb_sdk.wandb_run.Run = wandb.init(
            project=cfg.logger.project,
            name=cfg.logger.artifact_config.name,
            config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
            job_type="metrics",
        )  # type: ignore[assignment]
        cfg.preds_path = load_predictions(run=run, cfg=cfg)
    elif cfg.preds_path:
        cfg.preds_path = to_absolute_path(cfg.preds_path)
    else:
        raise ValueError("Either W&B artifact or local path should be provided to load predictions.")

    # ------------------------
    # -  aggregate metrics   -
    # ------------------------
    full_metrics = EvaluationMetrics(do_tensors=False, do_strings=True, prefix="test", shift=False)
    prefix_metrics = {
        i: EvaluationMetrics(n=i, do_tensors=False, do_strings=True, prefix="test", shift=False)
        for i in range(1, cfg.max_n_tokens + 1)
    }

    with jsonlines.open(cfg.preds_path, "r") as reader:
        for line in tqdm(reader, desc="Computing metrics"):
            pred = line["Prediction"].strip()
            target = line["Target"].strip()

            if not target:
                continue

            full_metrics.add_batch(
                predictions=[pred],
                references=[target],
            )

            pred_tokens = pred.split()
            target_tokens = target.split()

            for i in range(1, cfg.max_n_tokens + 1):
                if len(target_tokens) < i:
                    break

                pred_prefix_i = " ".join(pred_tokens[:i])
                target_prefix_i = " ".join(target_tokens[:i])
                prefix_metrics[i].add_batch(predictions=[pred_prefix_i], references=[target_prefix_i])

    # -----------------------
    # -   compute results   -
    # -----------------------
    full_metrics_results = full_metrics.compute()
    prefix_metrics_results = {}
    for i in prefix_metrics:
        try:
            prefix_metrics_results[i] = prefix_metrics[i].compute()
        except ValueError:
            logging.warning(f"Prefixes of length {i} did not appear in data")
        except ZeroDivisionError:
            logging.warning(f"ZeroDivisionError with prefixes of length {i}")

    for i in prefix_metrics_results:
        # we are using BLEU-4, ignore sequences of less than 4 tokens
        if i < 4:
            keys_to_drop = [key for key in prefix_metrics_results[i] if "b_norm" in key or "bleu" in key]

            for key in keys_to_drop:
                del prefix_metrics_results[i][key]

    # -----------------------
    # -      log results    -
    # -----------------------
    logging.info("Metrics for full sequences")
    logging.info(f"{full_metrics_results}")

    logging.info("Metrics for prefixes")
    logging.info(f"{prefix_metrics_results}")

    if cfg.logger.use_wandb:
        for i in prefix_metrics_results:
            wandb.log(prefix_metrics_results[i], step=i)
        wandb.log({f"{metric_name}_full": full_metrics_results[metric_name] for metric_name in full_metrics_results})


if __name__ == "__main__":
    main()
