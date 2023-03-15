from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class InputConfig:
    """
    Input configuration.

    Attributes:
        generate_with_history: `True` to concatenate commit message history with current commit message in decoder context during generation, `False` otherwise (ignored when `encoder_input_type` is `history`).
        train_with_history: `True` to concatenate commit message history with current commit message in decoder context during training, `False` otherwise (ignored when `encoder_input_type` is `history`).
        encoder_input_type: What type of input will be passed to encoder. Currently, `history` and `diff` are supported.
        context_ratio: A ratio of characters from input message to pass to model context during generation (should be in [0, 1] range).
    """

    generate_with_history: bool = True
    train_with_history: bool = MISSING
    encoder_input_type: str = MISSING
    context_ratio: float = 0.0
