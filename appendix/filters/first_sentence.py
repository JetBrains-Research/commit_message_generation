def is_one_sentence(text: str, nl_char: str = "\n") -> bool:
    """Implements single sentence filter.

    Args:
        text: Input string.

    Returns:
        True if input string has only one sentence, False otherwise.

    Notes:
        Determines the number of sentences based on newline characters.
    """
    lines = text.split(nl_char)
    return len(lines) == 1
