import logging
from pprint import pprint
from typing import List

import hydra
import pandas as pd
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils import EvaluationMetrics

logger = logging.getLogger("datasets")
logger.setLevel(logging.ERROR)


def get_tags(cfg: DictConfig) -> List[str]:
    tags = []
    if "transformer" in cfg.wandb.input_artifact_name:
        tags.append("transformer")
    elif "distilgpt2" in cfg.wandb.input_artifact_name:
        tags.append("distilgpt2")
    if "multilang" in cfg.wandb.output_artifact_type:
        tags.append("multilang")
    elif "java" in cfg.wandb.output_artifact_name:
        tags.append("java")
    if "with_history" in cfg.wandb.input_table_name:
        tags.append("generation with history")
    elif "without_history" in cfg.wandb.input_table_name:
        tags.append("generation without history")
    if "context_ratio" in cfg.wandb.input_table_name:
        tags.append(f"context_ratio = {cfg.wandb.input_table_name.split('context_ratio_')[-1]}")
    if "language" in cfg:
        tags.append(cfg.language)
        tags.append("single language")

    return tags


@hydra.main(config_path="conf", config_name="metrics_config")
def main(cfg: DictConfig):
    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")
    wandb.Table.MAX_ROWS = 50000

    if cfg.wandb:
        run = wandb.init(
            project=cfg.wandb.project, name=cfg.language if "language" in cfg else cfg.wandb.name, tags=get_tags(cfg)
        )
        input_table = run.use_artifact(cfg.wandb.input_artifact_name).get(cfg.wandb.input_table_name)
        df = pd.DataFrame(data=input_table.data, columns=input_table.columns)
        if "metadata_artifact_name" in cfg.wandb and "metadata_table_name" in cfg.wandb:
            metadata_table = run.use_artifact(cfg.wandb.metadata_artifact_name).get(cfg.wandb.metadata_table_name)
            metadata_df = pd.DataFrame(data=metadata_table.data, columns=metadata_table.columns)
            df = pd.concat([df, metadata_df], axis=1)
    else:
        df = pd.read_csv(to_absolute_path(cfg.input_file))

    if "language" in cfg:
        df = df.loc[df["language"] == cfg.language]

    full_metrics = EvaluationMetrics(do_tensors=False, do_strings=True, prefix="test")
    metrics = {
        i: EvaluationMetrics(n=i, do_tensors=False, do_strings=True, prefix="test")
        for i in range(1, cfg.max_n_tokens + 1)
    }

    sentence_level_metrics = []

    for _, row in tqdm(df[["Prediction", "Target"]].iterrows(), total=df.shape[0], desc="Computing metrics"):
        if not row["Target"].strip():
            continue

        cur_metrics = full_metrics.add_batch(
            predictions=[row["Prediction"].strip()], references=[row["Target"].strip()]
        )
        sentence_level_metrics.append({"edit_similarity": cur_metrics["edit_similarity"].item()})

        pred_words = row["Prediction"].strip().split()
        target_words = row["Target"].strip().split()

        for i in range(1, cfg.max_n_tokens + 1):
            if len(target_words) < i:
                break

            pred = " ".join(pred_words[:i])
            target = " ".join(target_words[:i])
            cur_metrics = metrics[i].add_batch(predictions=[pred], references=[target])

            if i in [1, 2, 5]:
                sentence_level_metrics[-1].update(
                    {
                        f"edit_similarity@{i}": cur_metrics["edit_similarity"].item(),
                        f"exact_match@{i}": cur_metrics["exact_match"].item(),
                    }
                )

    full_metrics = full_metrics.compute()
    metrics = {i: metrics[i].compute() for i in metrics}

    print("Metrics on full sequences")
    pprint(full_metrics)
    print()
    print("Metrics by number of tokens")
    pprint(metrics)

    if cfg.wandb:
        if "language" in cfg and cfg.language == "C++":
            suffix = "_cpp"
        elif "language" in cfg and cfg.language == "C#":
            suffix = "_csharp"
        else:
            suffix = f"_{cfg.language}" if "language" in cfg else ""
        for i in metrics:
            wandb.log(metrics[i], step=i)
        wandb.log({f"{metric_name}_full": full_metrics[metric_name] for metric_name in full_metrics})

        sentence_scores_df = pd.concat([df, pd.DataFrame(sentence_level_metrics)], axis=1)
        artifact = wandb.Artifact(name=f"{cfg.wandb.output_artifact_name}{suffix}", type=cfg.wandb.output_artifact_type)
        join_table = wandb.Table(dataframe=sentence_scores_df)
        artifact.add(join_table, f"{cfg.wandb.output_table_name}{suffix}")
        run.log_artifact(artifact)


if __name__ == "__main__":
    main()
