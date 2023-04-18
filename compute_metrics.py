import logging
import os
import random
from typing import Dict

import hydra
import jsonlines
import wandb
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from tqdm import tqdm

from conf import MetricsConfig
from src.utils import EvaluationMetrics

logger = logging.getLogger("datasets")
logger.setLevel(logging.ERROR)
random.seed(42)


def load_predictions(run: wandb.wandb_sdk.wandb_run.Run, cfg: MetricsConfig) -> str:
    input_artifact = run.use_artifact(
        f"{cfg.logger.artifact_config.project}/{cfg.logger.artifact_config.name}:{cfg.logger.artifact_config.version}"
    )
    if "tags" in input_artifact.metadata:
        run.tags = ["new_prefix_logic"] + (
            ["only_filtered" if cfg.filter.fit_filters else "only_unfiltered"] if cfg.filter.use_filtering else []
        )

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


def add_single_example(
    line: Dict[str, str], full_metrics: EvaluationMetrics, prefix_metrics: Dict[int, EvaluationMetrics]
) -> None:
    prediction = line["Prediction"].strip()
    target = line["Target"].strip()

    if not target:
        return

    full_metrics.add_batch(
        predictions=[prediction],
        references=[target],
    )

    pred_tokens = prediction.split()
    target_tokens = target.split()

    for i in prefix_metrics:
        pred_prefix_i = " ".join(pred_tokens[:i])
        target_prefix_i = " ".join(target_tokens[:i])
        prefix_metrics[i].add_batch(predictions=[pred_prefix_i], references=[target_prefix_i])


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
            job_type="metrics" if not cfg.filter.use_filtering else "filter_metrics",
        )  # type: ignore[assignment]
        cfg.preds_path = load_predictions(run=run, cfg=cfg)
    elif cfg.preds_path:
        cfg.preds_path = to_absolute_path(cfg.preds_path)
    else:
        raise ValueError("Either W&B artifact or local path should be provided to load predictions.")

    cfg.filter.path = to_absolute_path(cfg.filter.path)

    # ------------------------
    # -  aggregate metrics   -
    # ------------------------
    full_metrics = EvaluationMetrics(do_tensors=False, do_strings=True, prefix="test", shift=False)
    prefix_metrics = {
        i: EvaluationMetrics(n=i, do_tensors=False, do_strings=True, prefix="test", shift=False)
        for i in range(1, cfg.max_n_tokens + 1)
    }

    # default: simply compute the metrics for all the examples
    if not cfg.filter.use_filtering:
        with jsonlines.open(cfg.preds_path, "r") as reader:
            for line in tqdm(reader, desc="Computing metrics"):
                add_single_example(line, full_metrics=full_metrics, prefix_metrics=prefix_metrics)

    # or define filters configuration to control what subset will be considered
    else:
        def include_example(filters_line: Dict[str, str]) -> bool:
            """Combines all given filters via given logical operations and returns the final
            result: should we include the current example when calculating metrics or not."""
            if cfg.filter.logic == "and":
                result = all(filters_line[filter_col] for filter_col in cfg.filter.filters_to_include)
            elif cfg.filter.logic == "or":
                result = any(filters_line[filter_col] for filter_col in cfg.filter.filters_to_include)
            else:
                raise ValueError("`filter.logic` should be one of: `and`, `or`.")

            if cfg.filter.fit_filters:
                return result
            else:
                return not result

        # dry run: estimate the total number of examples and the number of examples in the filtered subset
        with jsonlines.open(cfg.preds_path, "r") as reader:
            total = sum(1 for _ in reader)
        with jsonlines.open(cfg.filter.path, "r") as filters_reader:
            included = sum(1 for filters_line in filters_reader if include_example(filters_line))

        # TODO: make configurable?
        # when computing metrics on out-of-filters subset, select a random subsample of the same size as filtered subset
        if not cfg.filter.fit_filters:
            num_examples_subset = total - included
            logging.warning(
                f"Total number of examples: {total}, will consider a random subset of {num_examples_subset} examples ({num_examples_subset / total * 100 :.2f}%)."
            )
            with jsonlines.open(cfg.filter.path, "r") as filters_reader:
                ids = [i for i, filters_line in enumerate(filters_reader) if include_example(filters_line)]
            subset_ids = set(random.sample(ids, k=num_examples_subset))
        else:
            subset_ids = None
            logging.warning(
                f"Total number of examples: {total}, will consider {included} examples ({included / total * 100 :.2f}%)."
            )

        with jsonlines.open(cfg.preds_path, "r") as reader:
            with jsonlines.open(cfg.filter.path, "r") as filters_reader:
                for i, (input_line, filters_line) in tqdm(
                    enumerate(zip(reader, filters_reader)), desc="Computing metrics with filters"
                ):
                    if (
                        not subset_ids
                        and include_example(filters_line)
                        or subset_ids
                        and i in subset_ids
                        and include_example(filters_line)
                    ):
                        add_single_example(input_line, full_metrics=full_metrics, prefix_metrics=prefix_metrics)

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
