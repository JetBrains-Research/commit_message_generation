import logging
from pprint import pprint

import hydra
import pandas as pd
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils import EvaluationMetrics, prepare_metrics_cfg

logger = logging.getLogger("datasets")
logger.setLevel(logging.ERROR)


def init_run(cfg: DictConfig) -> pd.DataFrame:
    cfg = prepare_metrics_cfg(cfg)
    run: wandb.wandb_sdk.wandb_run.Run = wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.model_name,
        job_type="metrics",
    )  # type: ignore[assignment]
    input_artifact = run.use_artifact(cfg.wandb.input_artifact_name)
    if "tags" in input_artifact.metadata:
        run.tags = input_artifact.metadata["tags"]
        if "language" in cfg and cfg.language:
            run.tags += ("single language",)
            run.tags += (cfg.language,)
    input_table = input_artifact.get(cfg.wandb.input_table_name)

    df = pd.DataFrame(data=input_table.data, columns=input_table.columns)
    if "metadata_artifact_name" in cfg.wandb and "metadata_table_name" in cfg.wandb:
        metadata_table = run.use_artifact(cfg.wandb.metadata_artifact_name).get(cfg.wandb.metadata_table_name)
        metadata_df = pd.DataFrame(data=metadata_table.data, columns=metadata_table.columns)

        # only happens when only first N examples were generated (e.g. for testing purposes)
        if len(df) < len(metadata_df):
            logging.warning(
                f"Predictions table is smaller than metadata table: {len(df)} vs {len(metadata_df)}! Trimming metadata table to {len(df)}"
            )
            metadata_df = metadata_df.head(len(df))
        df = pd.concat([df, metadata_df], axis=1)
    return df


@hydra.main(version_base=None, config_path="conf", config_name="metrics_config")
def main(cfg: DictConfig):
    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    if cfg.wandb:
        wandb.Table.MAX_ROWS = 50000
        df = init_run(cfg)
    else:
        df = pd.read_csv(to_absolute_path(cfg.input_file))

    if cfg.language:
        if "language" in df.columns:
            df = df.loc[df["language"] == cfg.language]
        else:
            logging.warning(
                f"Configured to evaluate only on {cfg.language} language, but metadata is not provided. Evaluating on full dataset"
            )

    if cfg.only_short_sequences:
        df = df.loc[df["bpe_num_tokens_diff"] <= 512]

    if cfg.only_long_sequences:
        df = df.loc[df["bpe_num_tokens_diff"] > 512]

    full_metrics = EvaluationMetrics(do_tensors=False, do_strings=True, prefix="test")
    prefix_metrics = {
        i: EvaluationMetrics(n=i, do_tensors=False, do_strings=True, prefix="test")
        for i in range(1, cfg.max_n_tokens + 1)
    }

    for _, row in tqdm(df[["Prediction", "Target"]].iterrows(), total=df.shape[0], desc="Computing metrics"):
        if not row["Target"].strip():
            continue

        full_metrics.add_batch(predictions=[row["Prediction"].strip()], references=[row["Target"].strip()])

        pred_words = row["Prediction"].strip().split()
        target_words = row["Target"].strip().split()

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

    if cfg.wandb:
        for i in prefix_metrics_results:
            wandb.log(prefix_metrics_results[i], step=i)
        wandb.log({f"{metric_name}_full": full_metrics_results[metric_name] for metric_name in full_metrics_results})


if __name__ == "__main__":
    main()
