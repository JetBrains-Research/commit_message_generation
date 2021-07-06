import hydra
from omegaconf import DictConfig
from .data import DataProcessor
from .model import GPT2Decoder


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    data_processor = DataProcessor(**cfg.data_processor)
    model = GPT2Decoder.from_pretrained(cfg.model.decoder_name_or_path)
    raise NotImplementedError()
