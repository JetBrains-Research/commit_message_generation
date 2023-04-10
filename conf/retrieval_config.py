from dataclasses import dataclass, field
from typing import Any, List

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
class SearchConfig:
    device: str = "cuda"
    load_index: bool = False
    index_root_dir: str = "faiss_indices"


@dataclass
class ArtifactRetrievalConfig:
    """
    Configuration for W&B artifact with model checkpoint.

    Artifact name is not provided because it's automatically retrieved from model and input configuration.

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
class WandbRetrievalConfig:
    """
    Configuration for W&B logging.

    What's logged during evaluation:
      * (optional) load model checkpoint from W&B artifact
      * model predictions

    Attributes:
        use_wandb: Whether W&B will be used for logging or not.
        project: Name of a project this run will appear in.
        use_api_key: True to read an API key from a local file (expected to be stored in `wandb_api_key.txt`).
        download_artifact: Whether model checkpoint should be downloaded from W&B artifact or not.
        input_artifact: Configuration for W&B artifact with model checkpoint.
        upload_artifact: Whether retrieved predictions should be uploaded to W&B artifact or not.
    """

    use_wandb: bool = True
    project: str = "commit_message_completion"
    use_api_key: bool = False
    download_artifact: bool = True
    input_artifact: ArtifactRetrievalConfig = field(default_factory=ArtifactRetrievalConfig)
    upload_artifact: bool = True


@dataclass
class EmbedderConfig:
    """
    Configuration for Transformer encoder that is used to construct embeddings.

    Args:
        device: Set to `cpu` to run model on CPU and `cuda` to run model on GPU. Currently, only single-GPU setting is supported; if your system has more than 1 GPU, make sure to set CUDA_VISIBLE_DEVICES enviromental variable to a single GPU.
        precision: Set to 16 to use native mixed precision from PyTorch.
        normalize_embeddings: Set to True to normalize embeddings, so that L2-norm is equal to 1.
    """

    device: str = "cpu"
    precision: int = 16
    normalize_embeddings: bool = True


@dataclass
class RetrievalConfig:
    """
    Configuration for retrieval.

    Args:
        ckpt_path: Local path to model checkpoint. Instead of this, you can also define a configuration for loading artifact at WandbEvalConfig.
    """

    defaults: List[Any] = field(default_factory=lambda: ["_self_", {"dataset": "multilang"}])

    ckpt_path: str = ""
    dataset: DatasetConfig = MISSING
    model: BaseModelConfig = MISSING
    input: InputConfig = field(default_factory=InputConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    embedder: EmbedderConfig = field(default_factory=EmbedderConfig)
    logger: WandbRetrievalConfig = field(default_factory=WandbRetrievalConfig)


cs = ConfigStore.instance()
cs.store(name="retrieval_config", node=RetrievalConfig)
cs.store(name="distilgpt2", group="model", node=DistilGPT2Config)
cs.store(name="random_roberta_2_random_gpt2_2", group="model", node=RandomTransformerConfig)
cs.store(name="codet5", group="model", node=CodeT5Config)
cs.store(name="codereviewer", group="model", node=CodeReviewerConfig)
cs.store(name="race", group="model", node=RACEConfig)
cs.store(name="multilang", group="dataset", node=DatasetConfig)
