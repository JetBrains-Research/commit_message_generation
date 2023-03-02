import json

import jsonlines

from src.data_utils.preprocessors import DefaultPreprocessor


def test_preprocess_mods():
    preprocessor = DefaultPreprocessor(diff_tokenizer=None, msg_tokenizer=None)  # tokenizers are not relevant

    # check that all mods types work correctly
    modify_mod = {
        "change_type": "MODIFY",
        "old_path": "fname",
        "new_path": "fname",
        "diff": "context 1[NL]context 2[NL]context 3[NL]-old line[NL]+new line[NL]",
    }
    assert preprocessor._preprocess_mods([modify_mod], line_sep="[NL]") == "fname[NL]" + modify_mod["diff"]

    add_mod = {
        "change_type": "ADD",
        "old_path": None,
        "new_path": "fname",
        "diff": "context 1[NL]context 2[NL]context 3[NL]-old line[NL]+new line[NL]",
    }
    assert preprocessor._preprocess_mods([add_mod], line_sep="[NL]") == "new file fname[NL]" + add_mod["diff"]

    delete_mod = {
        "change_type": "DELETE",
        "old_path": "fname",
        "new_path": None,
        "diff": "context 1[NL]context 2[NL]context 3[NL]-old line[NL]+new line[NL]",
    }
    assert preprocessor._preprocess_mods([delete_mod], line_sep="[NL]") == "deleted file fname[NL]" + delete_mod["diff"]

    rename_mod = {
        "change_type": "RENAME",
        "old_path": "fname1",
        "new_path": "fname2",
        "diff": "context 1[NL]context 2[NL]context 3[NL]-old line[NL]+new line[NL]",
    }
    assert (
        preprocessor._preprocess_mods([rename_mod], line_sep="[NL]")
        == "rename from fname1[NL]rename to fname2[NL]" + rename_mod["diff"]
    )

    copy_mod = {
        "change_type": "COPY",
        "old_path": "fname1",
        "new_path": "fname2",
        "diff": "context 1[NL]context 2[NL]context 3[NL]-old line[NL]+new line[NL]",
    }
    assert (
        preprocessor._preprocess_mods([copy_mod], line_sep="[NL]")
        == "copy from fname1[NL]copy to fname2[NL]" + copy_mod["diff"]
    )

    # check some mods together
    assert preprocessor._preprocess_mods([modify_mod, modify_mod, add_mod], line_sep="[NL]") == (
        "fname[NL]" + modify_mod["diff"] + "fname[NL]" + modify_mod["diff"] + "new file fname[NL]" + add_mod["diff"]
    )


def test_shuffle(tmp_path):
    preprocessor = DefaultPreprocessor(diff_tokenizer=None, msg_tokenizer=None)  # tokenizers are not relevant

    data = [{"hash": f"hash{i}", "data": f"row{i}"} for i in range(10)]
    with jsonlines.open(f"{tmp_path}/test_file.jsonl", "w") as writer:
        writer.write_all(data)

    retrieved_data = [{"sim": f"sim{i}", "data": f"row{i}"} for i in range(10)]
    with jsonlines.open(f"{tmp_path}/test_retrieved_file.jsonl", "w") as writer:
        writer.write_all(retrieved_data)

    for i in range(5):
        preprocessor._shuffle(f"{tmp_path}/test_file.jsonl", f"{tmp_path}/test_file_shuffled_{i}.jsonl")
        preprocessor._shuffle(
            f"{tmp_path}/test_retrieved_file.jsonl", f"{tmp_path}/test_retrieved_file_shuffled_{i}.jsonl"
        )
        with jsonlines.open(f"{tmp_path}/test_file_shuffled_{i}.jsonl", "r") as reader:
            shuffled_data = [line for line in reader]

        with jsonlines.open(f"{tmp_path}/test_retrieved_file_shuffled_{i}.jsonl", "r") as reader:
            retrieved_shuffled_data = [line for line in reader]

        assert shuffled_data != data
        assert [row["data"] for row in shuffled_data] == [row["data"] for row in retrieved_shuffled_data]


def test_get_pos_in_history():
    preprocessor = DefaultPreprocessor(diff_tokenizer=None, msg_tokenizer=None)  # tokenizers are not relevant
    positions = preprocessor._get_pos_in_history([1, 1, 2, 2, 3])
    assert positions == [0, 1, 0, 1, 0]
    assert preprocessor._num_commits == {1: 2, 2: 2, 3: 1}

    positions = preprocessor._get_pos_in_history([2, 1, 2, 55])
    assert positions == [2, 2, 3, 0]
    assert preprocessor._num_commits == {1: 3, 2: 4, 3: 1, 55: 1}


def test_process_history(tmp_path):
    preprocessor = DefaultPreprocessor(diff_tokenizer=None, msg_tokenizer=None)  # tokenizers are not relevant

    with jsonlines.open(f"{tmp_path}/test_file.jsonl", "w") as writer:
        writer.write_all(
            [{"author": i, "msg_input_ids": [i]} for i in range(10)]
            + [{"author": i, "msg_input_ids": [i + 100]} for i in range(5, 15)]
        )

    preprocessor._process_history(input_path=f"{tmp_path}/test_file.jsonl", output_path=f"{tmp_path}/test_history.json")
    with open(f"{tmp_path}/test_history.json", "r") as f:
        history = json.load(f)

    assert set(history.keys()) == set([f"{i}" for i in range(15)])
    for i in range(5):
        assert history[f"{i}"] == [[i]]
    for i in range(5, 10):
        assert history[f"{i}"] == [[i], [i + 100]]
    for i in range(10, 15):
        assert history[f"{i}"] == [[i + 100]]
