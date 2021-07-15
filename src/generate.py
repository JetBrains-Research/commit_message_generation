import omegaconf
import torch
from time import time
from typing import Dict, Optional, List, Union
from transformers.file_utils import ModelOutput
from .data_utils import DataProcessor
from .model import EncoderDecoder


def generate(
    model: EncoderDecoder,
    data_processor: DataProcessor,
    cfg: omegaconf.DictConfig,
    msg: str,
    history: List[str],
    crop_prompt: bool = False,
    diff: Optional[str] = None,
    encoder_outputs: Optional[ModelOutput] = None,
) -> Dict[str, Union[float, torch.Tensor]]:

    start_time = time()

    model_input = data_processor(msg=msg, history=history, diff=diff)

    processing_time = time()

    results, encoder_time, generation_time = model.generate(
        input_ids=model_input["decoder_input_ids"].to(cfg.device),
        encoder_input_ids=model_input["encoder_input_ids"].to(cfg.device),
        encoder_outputs=encoder_outputs,
        **cfg.generation_kwargs,
        min_length=cfg.min_length + model_input["decoder_input_ids"].shape[1],
        max_length=cfg.max_length + model_input["decoder_input_ids"].shape[1]
    )

    if crop_prompt:
        results["sequences"] = results["sequences"][:, model_input["decoder_input_ids"].shape[1]:]

    results["init_time"] = 0
    results["processing_time"] = processing_time - start_time
    results["encoder_time"] = encoder_time
    results["generation_time"] = generation_time
    results["encoder_input_shape"] = model_input["encoder_input_ids"].shape[1]
    results["decoder_input_shape"] = model_input["decoder_input_ids"].shape[1]
    return results
