import pytest
import omegaconf
from seq2seq_completion.api import ServerCMCApi
from seq2seq_completion.model import EncoderDecoder
from seq2seq_completion.data_utils import DataProcessor


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
def test_completion(default_test_setting, diff, msg, history):
    cfg, model, data_processor = default_test_setting
    ServerCMCApi._model = model
    ServerCMCApi._processor = data_processor
    ServerCMCApi.complete(diff=diff, msg=msg, history=history, **cfg.generation_kwargs)
