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
    "diff,decoder_context",
    [
        (
            "",
            "No diff and short message",
        ),
        (
            "- old \n + new",
            "Diff and short message",
        ),
        ("- old \n + new", "Hello" * 200 + " " + "Diff and long message"),
    ],
)
def test_completion(default_test_setting, diff, decoder_context):
    cfg, model, data_processor = default_test_setting
    ServerCMCApi._model = model
    ServerCMCApi._processor = data_processor
    ServerCMCApi.complete(diff=diff, decoder_context=decoder_context, **cfg.generation_kwargs)
