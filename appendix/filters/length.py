from nltk import wordpunct_tokenize


def is_shorter_than_n_tokens(text: str, n: int) -> bool:
    """Implements filter by length.

    Args:
        text: Input string.
        n: Number of tokens to consider.

    Returns:
        True if input string has <= n tokens, False otherwise.

    Notes:
        Performs tokenization by whitespaces and punctuation.
    """
    num_tokens = len(wordpunct_tokenize(text))
    return num_tokens <= n
