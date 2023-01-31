# This code is taken from replication package of "RACE: Retrieval-Augmented Commit Message Generation", EMNLP, 2022.
# https://github.com/DeepSoftwareAnalytics/RACE


import difflib

REPLACE_OLD = "<REPLACE_OLD>"
REPLACE_NEW = "<REPLACE_NEW>"
REPLACE_END = "<REPLACE_END>"

INSERT = "<INSERT>"
INSERT_OLD = "<INSERT_OLD>"
INSERT_NEW = "<INSERT_NEW>"
INSERT_END = "<INSERT_END>"

DELETE = "<DELETE>"
DELETE_END = "<DELETE_END>"

KEEP = "<KEEP>"
KEEP_END = "<KEEP_END>"


def compute_code_diffs(old_tokens, new_tokens):
    spans = []
    for edit_type, o_start, o_end, n_start, n_end in difflib.SequenceMatcher(
        None, old_tokens, new_tokens
    ).get_opcodes():
        if edit_type == "equal":
            spans.extend([KEEP] + old_tokens[o_start:o_end] + [KEEP_END])
        elif edit_type == "replace":
            spans.extend(
                [REPLACE_OLD] + old_tokens[o_start:o_end] + [REPLACE_NEW] + new_tokens[n_start:n_end] + [REPLACE_END]
            )
        elif edit_type == "insert":
            spans.extend([INSERT] + new_tokens[n_start:n_end] + [INSERT_END])
        else:
            spans.extend([DELETE] + old_tokens[o_start:o_end] + [DELETE_END])

    return spans
