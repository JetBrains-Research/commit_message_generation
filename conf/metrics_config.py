from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .data.dataset_config import DatasetConfig
from .data.input_config import InputConfig


@dataclass
class ArtifactMetricConfig:
    """
    Configuration for W&B artifact with model predictions.

    Args:
        project: W&B project.
        name: Name of W&B artifact.
        version: Version tag of W&B artifact (it should also match the table name in artifact!)
        local_path: Path to save artifact locally.
    """

    project: str = "saridormi/commit_message_completion"
    name: str = MISSING
    version: str = MISSING
    local_path: str = "artifacts"


@dataclass
class WandbMetricConfig:
    """
    Configuration for W&B logging.

    What's logged during metrics calculation:
      * (optional) load model predictions from W&B artifact
      * metrics

    Args:
        use_wandb: Whether W&B will be used for logging or not.
        project: Name of project this run will appear in.
        load_artifact: Whether model predictions should be loaded from W&B artifact or not.
    """

    use_wandb: bool = True
    project: str = "commit_message_completion"
    load_artifact: bool = True
    artifact_config: ArtifactMetricConfig = field(default_factory=ArtifactMetricConfig)


@dataclass
class FilterConfig:
    """
    Configuration for additional data filtering.

    Args:
        language: Pass language name to only calculate metrics on commits on this language.
        only_short_sequences: True to only calculate metrics on commits with diffs < 512 tokens.
        only_long_sequences: True to only calculate metrics on commits with diffs > 512 tokens.
    """

    language: Optional[str] = None
    only_short_sequences: Optional[bool] = None
    only_long_sequences: Optional[bool] = None


@dataclass
class MetricsConfig:
    """
    Configuration for metrics calculation.

    Metrics are calculated:
      * between full predictions and targets
      * between all prefixes of N tokens of predictions of targets

    Args:
        preds_path: Local path to model predictions. Instead of this, you can also define configuration for loading artifact at WandbMetricConfig.
        max_n_tokens: Maximum number of tokens (for prefix-level metrics).
    """

    defaults: List[Any] = field(default_factory=lambda: ["_self_", {"dataset": "multilang"}])

    preds_path: Optional[str] = None
    max_n_tokens: int = 15
    dataset: DatasetConfig = MISSING
    input: InputConfig = field(default_factory=InputConfig)
    logger: WandbMetricConfig = field(default_factory=WandbMetricConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)


cs = ConfigStore.instance()
cs.store(name="metrics_config", node=MetricsConfig)
cs.store(name="multilang", group="dataset", node=DatasetConfig)
