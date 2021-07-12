import omegaconf
import torch
from typing import Dict, Optional, List
from transformers.file_utils import ModelOutput
from .data_utils import DataProcessor
from .model import EncoderDecoder


def generate(
    config_path: str,
    msg: str,
    history: List[str],
    crop_prompt: bool = False,
    diff: Optional[str] = None,
    encoder_outputs: Optional[ModelOutput] = None,
) -> Dict[str, torch.Tensor]:
    cfg = omegaconf.OmegaConf.load(config_path)
    data_processor = DataProcessor(**cfg.data_processor)
    model = EncoderDecoder(**cfg.model).to(cfg.device)
    model_input = data_processor(msg=msg, history=history, diff=diff)
    cfg.generation_kwargs.min_length += model_input["decoder_input_ids"].shape[1]
    cfg.generation_kwargs.max_length += model_input["decoder_input_ids"].shape[1]

    results = model.generate(
        input_ids=model_input["decoder_input_ids"].to(cfg.device),
        encoder_input_ids=model_input["encoder_input_ids"].to(cfg.device),
        encoder_outputs=encoder_outputs,
        **cfg.generation_kwargs
    )

    if crop_prompt:
        results["sequences"] = results["sequences"][:, model_input["decoder_input_ids"].shape[1] :]
    return results
