from omegaconf import DictConfig
import torch
from typing import Dict, Optional, List
from transformers.file_utils import ModelOutput
from .data_utils import DataProcessor
from .model import EncoderDecoder


def generate(
    model: EncoderDecoder,
    data_processor: DataProcessor,
    cfg: DictConfig,
    msg: str,
    history: List[str],
    diff: Optional[str] = None,
    encoder_outputs: Optional[ModelOutput] = None,
) -> Dict[str, torch.Tensor]:

    # prepare input for generation
    model_input = data_processor(msg=msg, history=history, diff=diff)

    # generate
    results = model.generate(
        input_ids=model_input["decoder_input_ids"].to(cfg.device),
        encoder_input_ids=model_input["encoder_input_ids"].to(cfg.device),
        encoder_outputs=encoder_outputs,
        **cfg.generation_kwargs,
        min_length=cfg.min_length + model_input["decoder_input_ids"].shape[1],
        max_length=cfg.max_length + model_input["decoder_input_ids"].shape[1]
    )

    # remove prompt from generated tensors
    results["sequences"] = results["sequences"][:, model_input["decoder_input_ids"].shape[1] :]
    return results
