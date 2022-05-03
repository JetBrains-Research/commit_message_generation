import pytest
from src.metrics import EditSimilarity


@pytest.fixture
def edit_similarity():
    return EditSimilarity()


def test_same_string(edit_similarity):
    str1 = ["hello"]
    str2 = ["hello"]
    assert edit_similarity(str1, str2) == pytest.approx(1.0)


def test_empty_string(edit_similarity):
    str1 = [""]
    str2 = ["hello"]
    assert edit_similarity(str1, str2) == pytest.approx(0.0)


def test_both_empty(edit_similarity):
    str1 = [""]
    str2 = [""]
    assert edit_similarity(str1, str2).isnan().item()
