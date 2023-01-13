from src.data_utils.preprocessors import RACEPreprocessor


def test_preprocess_diff():
    preprocessor = RACEPreprocessor(diff_tokenizer=None, msg_tokenizer=None)  # tokenizers are not relevant
    assert (
        preprocessor._preprocess_diff(header=["fname"], diff="context[NL]-old line[NL]+new line[NL]", line_sep="[NL]")
        == "<KEEP> fname context <KEEP_END> <REPLACE_OLD> -old <REPLACE_NEW> +new <REPLACE_END> <KEEP> line <KEEP_END>"
    )

    assert (
        preprocessor._preprocess_diff(
            header=["new file fname"], diff="context[NL]-old line[NL]+new line[NL]", line_sep="[NL]"
        )
        == "<KEEP> new file fname context <KEEP_END> <REPLACE_OLD> -old <REPLACE_NEW> +new <REPLACE_END> <KEEP> line <KEEP_END>"
    )

    assert (
        preprocessor._preprocess_diff(
            header=["deleted file fname"], diff="context[NL]-old line[NL]+new line[NL]", line_sep="[NL]"
        )
        == "<KEEP> deleted file fname context <KEEP_END> <REPLACE_OLD> -old <REPLACE_NEW> +new <REPLACE_END> <KEEP> line <KEEP_END>"
    )

    assert (
        preprocessor._preprocess_diff(
            header=["rename from fname1", "rename to fname2"],
            diff="context[NL]-old line[NL]+new line[NL]",
            line_sep="[NL]",
        )
        == "<REPLACE_OLD> rename from fname1 <REPLACE_NEW> rename to fname2 <REPLACE_END> <KEEP> context <KEEP_END> <REPLACE_OLD> -old <REPLACE_NEW> +new <REPLACE_END> <KEEP> line <KEEP_END>"
    )

    assert (
        preprocessor._preprocess_diff(
            header=["copy from fname1", "copy to fname2"], diff="context[NL]-old line[NL]+new line[NL]", line_sep="[NL]"
        )
        == "<REPLACE_OLD> copy from fname1 <REPLACE_NEW> copy to fname2 <REPLACE_END> <KEEP> context <KEEP_END> <REPLACE_OLD> -old <REPLACE_NEW> +new <REPLACE_END> <KEEP> line <KEEP_END>"
    )
