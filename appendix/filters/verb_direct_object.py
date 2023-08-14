import spacy

spacy_nlp = spacy.load("en_core_web_sm")


def is_verb_direct_object(text: str):
    """Implements filter by Verb-Direct Object grammar structure via spaCy package.

    Args:
        text: Input string.

    Returns:
        True if input string starts with V-DO grammar structure, False otherwise.

    Notes:
        * Only the first sentence is considered.
        * Since past forms (e.g. fixed) and gerunds (e.g. fixing) are often not tagged as verbs,
          there is an extra preprocessing step: lemmatization of first word if it is a verb.
        * Current implementation supports not only Direct Objects consisting of single noun,
          but also clauses/phrases.
    """
    first_word = text.split(" ")[0]
    processed_first_word = spacy_nlp(first_word)[0]
    if processed_first_word.pos_ == "VERB":
        text = " ".join([processed_first_word.lemma_] + text.split(" ")[1:])

    doc = spacy_nlp(text)

    token = doc[0]
    if (
        token.pos_ == "VERB"
        and token.dep_ == "ROOT"
        and len([t.dep_ for t in token.children])
        and [t.dep_ for t in token.children][0] == "dobj"
    ):
        return True
    return False
