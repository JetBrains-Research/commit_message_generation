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
class OptimizerConfig:
    """
    Configuration for optimizer.

    Attributes:
        learning_rate: Learning rate for AdamW.
        initial_batch_size: If given, learning rate will be recalculated as (given lr) * (actual bs) / (initial bs).
        weight_decay: Weight decay for AdamW.
        num_warmup_steps: Number of warmup steps for linear scheduler with warmup.
        ratio_warmup_steps: Ratio of warmup steps for linear scheduler with warmup (so ratio_warmup_steps * total_steps will be used).
    """

    learning_rate: float = 1e-5
    initial_batch_size: Optional[int] = None
    weight_decay: float = 0.1
    num_warmup_steps: Optional[int] = None
    ratio_warmup_steps: Optional[float] = None


@dataclass
class ArtifactTrainConfig:
    """
    Configuration for W&B artifact.

    Artifact name is not configurable because it's automatically retrieved from model and input configuration.

    Attributes:
        load_artifact: True to download artifact from W&B, False otherwise.
        project: W&B project.
        version: Version tag of W&B artifact.
        artifact_path: Path to download in artifact.
    """

    load_artifact: bool = True
    project: str = "saridormi/commit_message_completion"
    version: str = "latest"
    artifact_path: str = "last.ckpt"


@dataclass
class WandbTrainConfig:
    """
    Configuration for W&B logging.

    What's logged during training:
      * loss & validation metrics
      * gradients
      * (optionally) model checkpoints

    Attributes:
        use_wandb: Whether W&B will be used for logging or not.
        use_api_key: True to read an API key from a local file (expected to be stored in `wandb_api_key.txt`).
        project: Name of a project this run will appear in.
        save_artifact: True to upload model checkpoints to W&B as artifacts, False otherwise.
        checkpoint: Artifact configuration for fine-tuned model checkpoint (option for RACE).
        retrieval: Artifact configuration for retrieved predictions (option for RACE).

    """

    use_wandb: bool = True
    use_api_key: bool = False
    project: str = "commit_message_completion"
    save_artifact: bool = True
    checkpoint: ArtifactTrainConfig = field(default_factory=ArtifactTrainConfig)
    retrieval: ArtifactTrainConfig = field(default_factory=ArtifactTrainConfig)


@dataclass
class TrainerTrainConfig:
    """
    Configuration for pytorch_lightning.Trainer. All options will be passed to Trainer as kwargs.

    Refer to docs: https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html

    Note:
        Defined fields are just the most frequently use arguments. You can easily add new ones using Hydra's
        override logic. E.g. `python train.py  ++trainer.devices=4 ++trainer.strategy=ddp`
    """

    max_epochs: int = 5
    precision: int = 16
    amp_backend: str = "native"
    accumulate_grad_batches: int = 1
    num_sanity_val_steps: int = 100
    gradient_clip_val: float = 1.0
    accelerator: str = "gpu"
    devices: Any = 1
    val_check_interval: Any = 1.0
    limit_train_batches: Optional[int] = None
    limit_val_batches: Optional[int] = None


@dataclass
class TrainConfig:
    """
    Configuration for training. For further information, refer to corresponding subconfig classes.
    """

    defaults: List[Any] = field(default_factory=lambda: ["_self_", {"dataset": "multilang"}])
    dataset: DatasetConfig = MISSING
    model: BaseModelConfig = MISSING
    input: InputConfig = field(default_factory=InputConfig)
    logger: WandbTrainConfig = field(default_factory=WandbTrainConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    trainer: TrainerTrainConfig = field(default_factory=TrainerTrainConfig)


cs = ConfigStore.instance()
cs.store(name="train_config", node=TrainConfig)
cs.store(name="distilgpt2", group="model", node=DistilGPT2Config)
cs.store(name="random_roberta_2_random_gpt2_2", group="model", node=RandomTransformerConfig)
cs.store(name="codet5", group="model", node=CodeT5Config)
cs.store(name="codereviewer", group="model", node=CodeReviewerConfig)
cs.store(name="race", group="model", node=RACEConfig)
cs.store(name="multilang", group="dataset", node=DatasetConfig)
