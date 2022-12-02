import json
import os
import random
import string
from itertools import combinations

import jsonlines
import numpy as np

from src.embedders import BagOfWordsEmbedder

np.random.seed(123)


def test_bow_embedder_fixed_vocab(tmp_path):
    embedder = BagOfWordsEmbedder(vocabulary={"aa": 0, "bb": 1})
    embedder._vectorizer.fit(["aa bb cc", "aa dd", "bb"])

    vocab_file = tmp_path / "vocab.json"
    vocab_file.touch()
    embedder.save_vocab(str(vocab_file))
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    assert vocab == embedder.vocab
    assert vocab == {"aa": 0, "bb": 1}

    assert all(embedder._transform(["aa bb cc"]).flatten() == [1, 1])
    assert all(embedder._transform(["cc"]).flatten() == [0, 0])
    assert all(embedder._transform(["xx"]).flatten() == [0, 0])


def test_bow_embedder_build_vocab(tmp_path):
    embedder = BagOfWordsEmbedder()
    embedder._vectorizer.fit(["aa bb cc", "aa dd", "bb"])

    vocab_file = tmp_path / "vocab.json"
    vocab_file.touch()
    embedder.save_vocab(str(vocab_file))
    with open(vocab_file, "r") as f:
        vocab = json.load(f)
    assert vocab == embedder.vocab
    assert vocab == {"aa": 0, "bb": 1, "cc": 2, "dd": 3}

    assert all(embedder._transform(["aa bb cc"]).flatten() == [1, 1, 1, 0])
    assert all(embedder._transform(["xx"]).flatten() == [0, 0, 0, 0])


def test_bow_embedder_build_vocab_several_calls():
    def simple_gen():
        a = iter((["xx", "aa yy"], ["aa bb cc", "aa dd", "bb"]))
        yield from (item for lst in a for item in lst)

    embedder = BagOfWordsEmbedder()
    embedder._vectorizer.fit(simple_gen())
    assert embedder.vocab == {"aa": 0, "bb": 1, "cc": 2, "dd": 3, "xx": 4, "yy": 5}


def test_bow_embedder_build_vocab_large(tmp_path):
    train_file = tmp_path / "input_data.jsonl"
    train_file.touch()

    temp_tokens = ["".join(tup) for tup in combinations(string.ascii_letters, 6)]
    print(f"\nMaximum possible vocabulary size: {len(temp_tokens)}")
    with jsonlines.open(str(train_file), "w") as f:
        for n_lines in range(10000):
            f.write(
                {
                    "mods": [
                        {
                            "change_type": "MODIFY",
                            "old_path": "path",
                            "new_path": "path",
                            "diff": " ".join(random.choices(temp_tokens, k=512)),
                        }
                    ]
                }
            )
    print(f"\nInput file size: {os.path.getsize(train_file) >> 20} MBs")

    embedder = BagOfWordsEmbedder(max_features=100000)
    embedder.build_vocab(input_filename=str(train_file), chunksize=10000)
