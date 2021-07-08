import pytest
from src.generate import generate


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
def test_generate(diff, msg, history):
    generate(config_path="configs/test_config.yaml", diff=diff, msg=msg, history=history)
