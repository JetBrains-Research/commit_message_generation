import pytest
import omegaconf
from src.generate import generate
from src.model import EncoderDecoder
from src.data_utils import DataProcessor


@pytest.fixture()
def default_test_setting():
    cfg = omegaconf.OmegaConf.load("configs/test_config.yaml")
    model = EncoderDecoder(**cfg.model).to(cfg.device)
    data_processor = DataProcessor(**cfg.data_processor)
    return cfg, model, data_processor


@pytest.mark.parametrize(
    "diff,msg,history",
    [
        (
            "",
            "Message only",
            [],
        ),
        (
            "",
            "No diff but history",
            ["history"],
        ),
        (
            "- old \n + new",
            "No history but diff",
            [""],
        ),
        (
            "- old \n + new",
            "All inputs",
            ["history"],
        ),
    ],
)
def test_generate(default_test_setting, diff, msg, history):
    cfg, model, data_processor = default_test_setting
    generate(cfg=cfg, model=model, data_processor=data_processor, diff=diff, msg=msg, history=history)
