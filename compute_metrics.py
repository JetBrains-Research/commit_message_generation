from pprint import pprint

import hydra
import pandas as pd
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils import EvaluationMetrics


@hydra.main(config_path="conf", config_name="metrics_config")
def main(cfg: DictConfig):
    print(f"==== Using config ====\n{OmegaConf.to_yaml(cfg)}")

    if cfg.wandb:
        run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)
        table = run.use_artifact(cfg.wandb.artifact_name).get(cfg.wandb.table_name)
        df = pd.DataFrame(data=table.data, columns=table.columns)
    else:
        df = pd.read_csv(to_absolute_path(cfg.input_file))

    full_metrics = EvaluationMetrics(do_tensors=False, do_strings=True, prefix="test")
    metrics = {
        i: EvaluationMetrics(n=i, do_tensors=False, do_strings=True, prefix="test")
        for i in range(1, cfg.max_n_tokens + 1)
    }

    for _, row in tqdm(df[["Prediction", "Target"]].iterrows(), total=df.shape[0], desc="Computing metrics"):
        full_metrics.add_batch(predictions=[row["Prediction"].strip()], references=[row["Target"].strip()])

        pred_words = row["Prediction"].strip().split()
        target_words = row["Target"].strip().split()
        for i in range(1, cfg.max_n_tokens + 1):
            if len(target_words) < i:
                break

            pred = " ".join(pred_words[:i])
            target = " ".join(target_words[:i])
            metrics[i].add_batch(predictions=[pred], references=[target])

    full_metrics = full_metrics.compute()
    metrics = {i: metrics[i].compute() for i in metrics}

    print("Metrics on full sequences")
    pprint(full_metrics)
    print()
    print("Metrics by number of tokens")
    pprint(metrics)

    if cfg.wandb:
        for i in metrics:
            wandb.log(metrics[i], step=i)
        wandb.log({f"{metric_name}_full": full_metrics[metric_name] for metric_name in full_metrics})


if __name__ == "__main__":
    main()
