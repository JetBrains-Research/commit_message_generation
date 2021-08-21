import pytest
from seq2seq_completion.api.post_processing_utils import PostProcessor


@pytest.mark.parametrize(
    "str_before,expected_str_after",
    [
        ("no punctuation nothing should change", "no punctuation nothing should change"),
        ("docker - java", "docker-java"),
        ("version 1 . 0 . 0", "version 1.0.0"),
        ("ugly case : some sentence . some other sentence", "ugly case: some sentence.some other sentence"),
        (
            "we need it to support e . g . ` filename . py ` , which is more likely to appear in commit messages",
            "we need it to support e.g. `filename.py`, which is more likely to appear in commit messages",
        ),
        (
            "ugly case : random commit message ( some additional explanation )",
            "ugly case: random commit message(some additional explanation)",
        ),
        (
            "we need it to support e . g . ` function ( args ) ` , which is more likely to appear in commit messages",
            "we need it to support e.g. `function(args)`, which is more likely to appear in commit messages",
        ),
        ("Fix bug , do smth else", "Fix bug, do smth else"),
        (
            "[ # 123 ] Refer to GitHub issue . . . \n See also : # 456 , # 789",
            "[#123] Refer to GitHub issue...\nSee also: #456, #789",
        ),
        ("Merge branch ' branch ' ", "Merge branch 'branch'"),
        ('Revert commit " commit " ', 'Revert commit "commit"'),
        ("merge pull request # 123 from user / branch", "merge pull request #123 from user/branch"),
        (
            "https : / / pytorch - lightning . readthedocs . io / en / latest / common / trainer . html # test",
            "https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#test",
        ),
        ("user @ mail . com", "user@mail.com"),
    ],
)
def test_small_examples(str_before: str, expected_str_after: str):
    assert PostProcessor.process(str_before) == expected_str_after


@pytest.mark.parametrize(
    "str_before,expected_str_after",
    [
        (
            "Simplify serialize . h ' s exception handling \n Remove the ' state ' and ' exceptmask ' from serialize . h's "
            "stream implementations, as well as related methods . \n As exceptmask always included ' failbit ' , "
            "and setstate was always called with bits = failbit , \n all it did was immediately raise an exception . Get "
            "rid of those variables , \n and replace the setstate with direct exception throwing ( which also removes "
            "some dead code ) . \n As a result , good ( ) is never reached after a failure ( there are only 2 calls , "
            "one of which is in tests ) , \n and can just be replaced by ! eof ( ) . fail ( ) , clear ( n ) and "
            "exceptions ( ) are just never called . Delete them . ",
            "Simplify serialize.h's exception handling\nRemove the 'state' and 'exceptmask' from serialize.h's stream "
            "implementations, as well as related methods.\nAs exceptmask always included 'failbit', and setstate was "
            "always called with bits=failbit,\nall it did was immediately raise an exception.Get rid of those variables,"
            "\nand replace the setstate with direct exception throwing(which also removes some dead code).\nAs a result, "
            "good() is never reached after a failure(there are only 2 calls, one of which is in tests),\nand can just be "
            "replaced by! eof().fail(), clear(n) and exceptions() are just never called.Delete them.",
        )
    ],
)
def test_big_examples(str_before: str, expected_str_after: str):
    assert PostProcessor.process(str_before) == expected_str_after
