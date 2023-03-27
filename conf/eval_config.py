from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .data.dataset_config import DatasetConfig
from .data.input_config import InputConfig
from .model.base_configs import BaseModelConfig
from .model.configs import (
    CodeReviewerConfig,
    CodeT5Config,
    DistilGPT2Config,
    RACEConfig,
    RandomTransformerConfig,
)


@dataclass
class ArtifactEvalConfig:
    """
    Configuration for W&B artifact with model checkpoint.

    Artifact name is not provided, because it's automatically retrieved from model and input configuration.

    Attributes:
        project: W&B project.
        version: Version tag of W&B artifact.
        artifact_path: Path to model checkpoint in artifact.
        local_path: Path to save artifact locally.
    """

    project: str = "saridormi/commit_message_completion"
    version: str = "latest"
    artifact_path: str = "last.ckpt"
    local_path: str = "artifacts"


@dataclass
class WandbEvalConfig:
    """
    Configuration for W&B logging.

    What's logged during evaluation:
      * (optional) load model checkpoint from W&B artifact
      * model predictions

    Attributes:
        use_wandb: Whether W&B will be used for logging or not.
        project: Name of project this run will appear in.
        load_artifact: Whether model checkpoint should be loaded from W&B artifact or not.
        use_api_key: True to read an API key from a local file (expected to be stored in `wandb_api_key.txt`).
    """

    use_wandb: bool = True
    project: str = "commit_message_completion"
    load_artifact: bool = True
    use_api_key: bool = False
    artifact_config: ArtifactEvalConfig = field(default_factory=ArtifactEvalConfig)


@dataclass
class TrainerEvalConfig:
    """
    Configuration for pytorch_lightning.Trainer. All options will be passes to Trainer as kwargs.
    (refer to docs: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html)
    """

    accelerator: str = "gpu"
    devices: Any = 1
    limit_test_batches: Optional[int] = None


@dataclass
class GenerationConfig:
    """
    Configuration for generation.

    All options will be passed to HuggingFace's generate() as kwargs.
    (refer to docs: https://huggingface.co/docs/transformers/main_classes/text_generation)
    """

    num_beams: int = 10
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    max_new_tokens: int = 15


@dataclass
class EvalConfig:
    """
    Configuration for evaluation.

    Args:
        stage: Set to "sweep" if you want to use validation data for tuning hyperparameters.
        ckpt_path: Local path to model checkpoint. Instead of this, you can also define configuration for loading artifact at WandbEvalConfig.
    """

    defaults: List[Any] = field(default_factory=lambda: ["_self_", {"dataset": "multilang"}])

    stage: str = "test"
    ckpt_path: str = ""
    dataset: DatasetConfig = MISSING
    model: BaseModelConfig = MISSING
    input: InputConfig = field(default_factory=InputConfig)
    logger: WandbEvalConfig = field(default_factory=WandbEvalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    trainer: TrainerEvalConfig = field(default_factory=TrainerEvalConfig)


cs = ConfigStore.instance()
cs.store(name="eval_config", node=EvalConfig)
cs.store(name="distilgpt2", group="model", node=DistilGPT2Config)
cs.store(name="random_roberta_2_random_gpt2_2", group="model", node=RandomTransformerConfig)
cs.store(name="codet5", group="model", node=CodeT5Config)
cs.store(name="codereviewer", group="model", node=CodeReviewerConfig)
cs.store(name="race", group="model", node=RACEConfig)
cs.store(name="multilang", group="dataset", node=DatasetConfig)
