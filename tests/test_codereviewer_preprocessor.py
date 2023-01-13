from src.data_utils.preprocessors import CodeReviewerPreprocessor


def test_preprocess_diff():
    preprocessor = CodeReviewerPreprocessor(diff_tokenizer=None, msg_tokenizer=None)  # tokenizers are not relevant

    assert (
        preprocessor._preprocess_diff(
            "context 1[NL]context 2[NL]context 3[NL]-old line[NL]+new line[NL]", line_sep="[NL]"
        )
        == "<keep>context 1[NL]<keep>context 2[NL]<keep>context 3[NL]<del>old line[NL]<add>new line[NL]"
    )
