from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


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
        max_n_tokens: Maximum number of tokens (for prefix-level metrics).
    """

    preds_path: Optional[str] = None
    max_n_tokens: int = 15
    logger: WandbMetricConfig = field(default_factory=WandbMetricConfig)


cs = ConfigStore.instance()
cs.store(name="metrics_config", node=MetricsConfig)
