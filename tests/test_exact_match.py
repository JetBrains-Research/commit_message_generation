import pytest
from src.metrics import ExactMatch


@pytest.mark.parametrize(
    "str1,str2,n",
    [
        (["hello\n\n\n"], ["hello world"], 1),
        (["full match"], ["full match"], 2),
        (["a a a a a x y z"], ["a a a a a b c d"], 5),
    ],
)
def test_full_match(str1, str2, n):
    assert ExactMatch(n=n)(str1, str2) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "str1,str2,n",
    [
        (["random words"], ["something else then random words again"], 2),
        ([""], ["anything"], 1),
        (["anything"], [""], 1),
    ],
)
def test_no_match(str1, str2, n):
    assert ExactMatch(n=n)(str1, str2) == pytest.approx(0.0)


def test_random_example():
    str1 = ["a b c d"]
    str2 = ["a x y d"]
    assert ExactMatch(n=1)(str1, str2) == pytest.approx(1.0)
    assert ExactMatch(n=2)(str1, str2) == pytest.approx(1 / 2)
    assert ExactMatch(n=3)(str1, str2) == pytest.approx(1 / 3)
    assert ExactMatch(n=4)(str1, str2) == pytest.approx(1 / 2)
