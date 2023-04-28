from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    """Configuration for data preprocessing.

    Attributes:
        root_dir: Path to root directory with input data.
        input_path: Filename of input file (should be located in root directory).
        input_path_unfinished_preds: Path to file with unfinished predictions (optional, necessary only to complete
         unfinished run).
        max_number_of_tokens: Maximum allowed number of BPE tokens in diffs (they will be truncated to this number).
        prompt_configuration: A type of prompt constructor to use.
          Currently supported: `simple`, `history`.
        use_cache: True to reuse existing files when found, False otherwise.
        chunksize: Number of examples in a single chunk.
    """

    root_dir: str = "raw_data/multilang"
    input_path: str = "test.jsonl"
    input_path_unfinished_preds: Optional[str] = None
    max_number_of_tokens: int = 512
    prompt_configuration: str = MISSING
    use_cache: bool = False
    chunksize: int = 1000


@dataclass
class WandbConfig:
    """
    Configuration for W&B logging.

    What's logged during evaluation:
      * model predictions

    Attributes:
        use_wandb: Whether W&B will be used for logging or not.
        project: Name of a project this run will appear in.
        use_api_key: True to read an API key from a local file (expected to be stored in `wandb_api_key.txt`).
        upload_artifact: Whether predictions should be uploaded to W&B artifact or not.
    """

    use_wandb: bool = True
    project: str = "commit_message_completion"
    use_api_key: bool = False
    upload_artifact: bool = True


@dataclass
class GenerationConfig:
    """
    Generation parameters for OpenAI's Completion & ChatCompletion endpoints.

    Relevant resources:
      * Completion: https://platform.openai.com/docs/api-reference/completions/create
      * Chat Completion: https://platform.openai.com/docs/api-reference/chat/create
    """

    max_tokens: int = 15
    temperature: float = 0.8
    top_p: float = 0.95
    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


@dataclass
class OpenAIConfig:
    """Configuration for experiments with OpenAI models."""

    model_id: str = MISSING
    context_ratio: float = 0.0
    fill_file: bool = False
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    logger: WandbConfig = field(default_factory=WandbConfig)


cs = ConfigStore.instance()
cs.store(name="openai_config", node=OpenAIConfig)
