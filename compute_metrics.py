import logging
from pprint import pprint
import os
import hydra
import pandas as pd
import wandb
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

from conf import MetricsConfig
from src.utils import EvaluationMetrics

logger = logging.getLogger("datasets")
logger.setLevel(logging.ERROR)


def load_preidctions(run: wandb.wandb_sdk.wandb_run.Run, cfg: MetricsConfig) -> pd.DataFrame:
    input_artifact = run.use_artifact(
        f"{cfg.logger.artifact_config.project}/{cfg.logger.artifact_config.name}:{cfg.logger.artifact_config.version}"
    )
    if "tags" in input_artifact.metadata:
        run.tags = input_artifact.metadata["tags"]
        if cfg.filter.language:
            run.tags += ("single language",)
            run.tags += (cfg.filter.language,)
    input_table: wandb.Table = input_artifact.get(cfg.logger.artifact_config.version)  # type: ignore[assignment]
    df = pd.DataFrame(data=input_table.data, columns=input_table.columns)
    return df


@hydra.main(version_base="1.1", config_path="conf", config_name="metrics_config")
def main(cfg: MetricsConfig):
    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    if cfg.logger.use_wandb:
        if cfg.logger.use_api_key:
            with open(hydra.utils.to_absolute_path("wandb_api_key.txt"), "r") as f:
                os.environ["WANDB_API_KEY"] = f.read().strip()
        wandb.Table.MAX_ROWS = 50000
        run: wandb.wandb_sdk.wandb_run.Run = wandb.init(
            project=cfg.logger.project,
            name=cfg.logger.artifact_config.name,
            job_type="metrics",
        )  # type: ignore[assignment]
        df = load_preidctions(run=run, cfg=cfg)
    elif cfg.preds_path:
        df = pd.read_csv(to_absolute_path(cfg.preds_path))
    else:
        raise ValueError("Predictions should be either loaded from W&B artifact or from local file.")

    if cfg.filter.language:
        if "language" in df.columns:
            df = df.loc[df["language"] == cfg.filter.language]
        else:
            logging.warning(
                f"Configured to evaluate only on {cfg.filter.language} language, but metadata is not provided. Evaluating on full dataset"
            )

    if cfg.filter.only_short_sequences:
        if "bpe_num_tokens_diff" in df.columns:
            df = df.loc[df["bpe_num_tokens_diff"] <= 512]
        else:
            logging.warning(
                f"Configured to evaluate only on short sequences, but metadata is not provided. Evaluating on full dataset"
            )

    if cfg.filter.only_long_sequences:
        if "bpe_num_tokens_diff" in df.columns:
            df = df.loc[df["bpe_num_tokens_diff"] > 512]
        else:
            logging.warning(
                f"Configured to evaluate only on long sequences, but metadata is not provided. Evaluating on full dataset"
            )

    full_metrics = EvaluationMetrics(do_tensors=False, do_strings=True, prefix="test", shift=False)
    prefix_metrics = {
        i: EvaluationMetrics(n=i, do_tensors=False, do_strings=True, prefix="test", shift=False)
        for i in range(1, cfg.max_n_tokens + 1)
    }

    for _, row in tqdm(df[["Prediction", "Target"]].iterrows(), total=df.shape[0], desc="Computing metrics"):
        if not row["Target"].strip():
            continue

        full_metrics.add_batch(
            predictions=[row["Prediction"].strip().replace("[NL]", "\n")],
            references=[row["Target"].strip().replace("[NL]", "\n")],
        )

        pred_words = row["Prediction"].strip().replace("[NL]", "\n").split()
        target_words = row["Target"].strip().replace("[NL]", "\n").split()

        for i in range(1, cfg.max_n_tokens + 1):
            if len(target_words) < i:
                break

            pred = " ".join(pred_words[:i])
            target = " ".join(target_words[:i])
            prefix_metrics[i].add_batch(predictions=[pred], references=[target])

    full_metrics_results = full_metrics.compute()
    prefix_metrics_results = {}
    for i in prefix_metrics:
        try:
            prefix_metrics_results[i] = prefix_metrics[i].compute()
        except ValueError:
            logging.warning(f"Prefixes of length {i} did not appear in data")
        except ZeroDivisionError as e:
            logging.warning(f"ZeroDivisionError with prefixes of length {i}")

    print("Metrics on full sequences")
    pprint(full_metrics_results)
    print()
    print("Metrics by number of tokens")
    pprint(prefix_metrics_results)

    if cfg.logger.use_wandb:
        for i in prefix_metrics_results:
            wandb.log(prefix_metrics_results[i], step=i)
        wandb.log({f"{metric_name}_full": full_metrics_results[metric_name] for metric_name in full_metrics_results})


if __name__ == "__main__":
    main()
