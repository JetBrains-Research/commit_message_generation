import pytest

from src.metrics import EditSimilarity


@pytest.fixture
def edit_similarity():
    return EditSimilarity()


@pytest.mark.parametrize(
    "input_str",
    [("hello"), ('@pytest.mark.parametrize(\n   "input_str",'), ("def test_same_string(edit_similarity, input_str):")],
)
def test_same_string(edit_similarity, input_str):
    assert edit_similarity([input_str], [input_str]) == pytest.approx(1.0)


def test_empty_pred(edit_similarity):
    assert edit_similarity([""], ["hello"]) == pytest.approx(0.0)


def test_empty_ref(edit_similarity):
    assert edit_similarity([""], [""]).isnan().item()
    assert edit_similarity(["hello"], [""]).isnan().item()
