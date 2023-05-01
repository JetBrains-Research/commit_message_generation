from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class FilterConfig:
    """
    Configuration for additional data filtering when calculating metrics.

    Attributes:
        path: Path to file with filters metadata for a test set.
        use_filtering: True to use additional data filtering, False otherwise.
        filters_to_include: List of column names to consider. Each column should be boolean.
        logic: A logic to follow when multiple columns are given (`and` for logical and, `or` for logical or).
        fit_filters: If True, will consider examples that fit given columns with given logic.
          If False, will consider examples that DON'T FIT given columns with given logic.
        use_pos_in_file_filtering: True to use `pos_in_file` column and only consider lines present in a given file,
          False to use boolean filters logic.

    """

    path: str = "raw_data/multilang/downsample/filters/test.jsonl"
    use_filtering: bool = False
    filters_to_include: List[str] = field(
        default_factory=lambda: ["is_vdo", "one_sentence_newline", "message_30_tokens", "diff_100_tokens"]
    )
    logic: str = "and"
    fit_filters: bool = True
    use_pos_in_file_filtering: bool = False
    use_subset: bool = False
    subset_num_examples: Optional[int] = None


@dataclass
class ArtifactMetricConfig:
    """
    Configuration for W&B artifact with model predictions.

    Attributes:
        project: W&B project.
        name: Name of W&B artifact.
        version: Version tag of W&B artifact.
        artifact_path: Path to model predictions in artifact.
        local_path: Path to save artifact locally.
    """

    project: str = "saridormi/commit_message_completion"
    name: str = MISSING
    version: str = "latest"
    artifact_path: str = MISSING
    local_path: str = "artifacts"


@dataclass
class WandbMetricConfig:
    """
    Configuration for W&B logging.

    What's logged during metrics calculation:
      * (optional) load model predictions from W&B artifact
      * metrics

    Attributes:
        use_wandb: Whether W&B will be used for logging or not.
        project: Name of project this run will appear in.
        load_artifact: Whether model predictions should be loaded from W&B artifact or not.
        use_api_key: True to read an API key from a local file (expected to be stored in `wandb_api_key.txt`).
    """

    use_wandb: bool = True
    project: str = "commit_message_completion"
    load_artifact: bool = True
    use_api_key: bool = False
    artifact_config: ArtifactMetricConfig = field(default_factory=ArtifactMetricConfig)


@dataclass
class MetricsConfig:
    """
    Configuration for metrics calculation.

    Metrics are calculated:
      * between full predictions and targets
      * between all prefixes of N tokens of predictions of targets

    Attributes:
        preds_path: Local path to model predictions. Instead of this, you can also define configuration for loading artifact at WandbMetricConfig.
        include_short: False to only consider messages with >= i tokens when computing metrics for prefixes of i tokens,
         True to include all messages.
        max_n_tokens: Maximum number of tokens (for prefix-level metrics).
    """

    preds_path: Optional[str] = None
    include_short: bool = False
    max_n_tokens: int = 15
    filter: FilterConfig = field(default_factory=FilterConfig)
    logger: WandbMetricConfig = field(default_factory=WandbMetricConfig)


cs = ConfigStore.instance()
cs.store(name="metrics_config", node=MetricsConfig)
