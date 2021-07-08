import omegaconf
import torch
from typing import Dict, Optional, List
from transformers.file_utils import ModelOutput  # type: ignore
from .data_utils import DataProcessor
from .model import EncoderDecoder


def generate(
    config_path: str,
    msg: str,
    history: List[str],
    diff: Optional[str] = None,
    encoder_outputs: Optional[ModelOutput] = None,
) -> Dict[str, torch.Tensor]:
    cfg = omegaconf.OmegaConf.load(config_path)
    data_processor = DataProcessor(**cfg.data_processor)
    model = EncoderDecoder(**cfg.model)
    model_input = data_processor(msg=msg, history=history, diff=diff)
    return model.generate(
        input_ids=model_input["decoder_input_ids"],
        encoder_input_ids=model_input["encoder_input_ids"],
        encoder_outputs=encoder_outputs,
        **cfg.generation_kwargs
    )
