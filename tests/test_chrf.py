import pytest
from src.metrics import ChrF


# took example from here: https://www.nltk.org/_modules/nltk/translate/chrf_score.html#corpus_chrf
def test_nltk_example():
    ref1 = "It is a guide to action that ensures that the military will forever heed Party commands".split()
    ref2 = "It is the guiding principle which guarantees the military forces always being under the command of the Party".split()

    hyp1 = "It is a guide to action which ensures that the military always obeys the commands of the party".split()
    hyp2 = "It is to insure the troops forever hearing the activity guidebook that party direct".split()

    chrf = ChrF()
    chrf.add_batch(references=[ref1, ref2, ref1, ref2], predictions=[hyp1, hyp2, hyp2, hyp1])
    assert chrf.compute()["chrf"] == pytest.approx(0.391, abs=1e-4)

    chrf = ChrF()
    chrf.add_batch(references=[ref1, ref2], predictions=[hyp1, hyp2])
    chrf.add_batch(references=[ref1, ref2], predictions=[hyp2, hyp1])
    assert chrf.compute()["chrf"] == pytest.approx(0.391, abs=1e-4)

    chrf = ChrF()
    chrf.add_batch(references=[ref1], predictions=[hyp1])
    chrf.add_batch(references=[ref2], predictions=[hyp2])
    chrf.add_batch(references=[ref1], predictions=[hyp2])
    chrf.add_batch(references=[ref2], predictions=[hyp1])
    assert chrf.compute()["chrf"] == pytest.approx(0.391, abs=1e-4)


# took examples from here: https://github.com/nltk/nltk/blob/develop/nltk/translate/chrf_score.py#L39
@pytest.mark.parametrize(
    "ref,hyp,result",
    [
        (
            "It is a guide to action that ensures that the military will forever heed Party commands".split(),
            "It is a guide to action which ensures that the military always obeys the commands of the party".split(),
            0.6349,
        ),
        (
            "It is a guide to action that ensures that the military will forever heed Party commands".split(),
            "It is to insure the troops forever hearing the activity guidebook that party direct".split(),
            0.3330,
        ),
        ("the cat is on the mat".split(), "the the the the the the the".split(), 0.1468),
    ],
)
def test_nltk_sentence_examples(ref, hyp, result):
    chrf = ChrF()
    chrf.add_batch(references=[ref], predictions=[hyp])
    assert chrf.compute()["chrf"] == pytest.approx(result, abs=1e-4)
