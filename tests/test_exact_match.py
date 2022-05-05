import pytest

from src.metrics import ExactMatch


@pytest.mark.parametrize(
    "predictions,references,n",
    [
        (["hello\n\n\n"], ["hello world"], 1),
        (["full match"], ["full match"], 2),
        (["a a a a a x y z"], ["a a a a a b c d"], 5),
    ],
)
def test_full_match(predictions, references, n):
    assert ExactMatch(n=n)(predictions, references) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "predictions,references,n",
    [
        (["random words"], ["something else then random words again"], 2),
        ([""], ["anything"], 1),
    ],
)
def test_no_match(predictions, references, n):
    assert ExactMatch(n=n)(predictions, references) == pytest.approx(0.0)


def test_different_lengths():
    predictions = ["a b c", "d f"]
    references = ["a b c", "d e"]
    assert ExactMatch(n=1)(predictions, references) == pytest.approx(1.0)
    assert ExactMatch(n=2)(predictions, references) == pytest.approx(0.5)
    assert ExactMatch(n=3)(predictions, references) == pytest.approx(1.0)

    for n in range(4, 100):
        assert ExactMatch(n=n)(predictions, references) == pytest.approx(0.0)
