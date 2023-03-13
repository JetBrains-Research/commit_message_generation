import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torchmetrics import MetricCollection
from torchmetrics.utilities import check_forward_full_state_property
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from src.metrics import Accuracy

pl.seed_everything(123)


@pytest.fixture
def metrics_collection():
    return MetricCollection(
        {
            "accuracy@1": Accuracy(top_k=1, ignore_index=-100),
            "accuracy@2": Accuracy(top_k=2, ignore_index=-100),
            "accuracy@3": Accuracy(top_k=3, ignore_index=-100),
        }
    )


@pytest.fixture
def metrics_collection_no_shift():
    return MetricCollection(
        {
            "accuracy@1": Accuracy(top_k=1, shift=False, ignore_index=-100),
            "accuracy@2": Accuracy(top_k=2, shift=False, ignore_index=-100),
            "accuracy@3": Accuracy(top_k=3, shift=False, ignore_index=-100),
        }
    )


def test_full_top1_match(metrics_collection):
    scores = torch.tensor(
        [[0.5, 0, 0.3, 0.2, 0.4], [1.0, -100, -200, -101, 0], [-1, -1, -1, -1, -1]], dtype=torch.float
    )
    labels = torch.tensor([-1, 0, 0], dtype=torch.long)
    results = metrics_collection(scores, labels)
    assert np.allclose([results["accuracy@1"].item(), results["accuracy@3"].item()], [1.0, 1.0], rtol=1e-05, atol=1e-08)


def test_full_top3_match(metrics_collection):
    scores = torch.tensor(
        [[0.5, 0, 0.3, 0.2, 0.4], [1.0, -100, -200, -101, 0], [-1, -1, -1, -1, -1]], dtype=torch.float
    )
    labels = torch.tensor([-1, 2, 1], dtype=torch.long)
    results = metrics_collection(scores, labels)
    assert np.allclose([results["accuracy@1"].item(), results["accuracy@3"].item()], [0.0, 1.0], rtol=1e-05, atol=1e-08)


def test_different_batch_sizes(metrics_collection):
    # batch size 4
    scores = torch.tensor(
        [
            [[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3], [-1, -1, -1]],
            [[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3], [-1, -1, -1]],
            [[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3], [-1, -1, -1]],
            [[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3], [-1, -1, -1]],
        ],
        dtype=torch.float,
    )
    labels = torch.tensor([[-1, 0, 2, -100], [-1, 1, 1, -100], [-1, 2, 1, 3], [-1, 1, 2, -100]], dtype=torch.long)
    results = metrics_collection(scores, labels)
    assert np.allclose(
        [results["accuracy@1"].item(), results["accuracy@2"].item()], [0.5 / 4, (1 + 2 / 3) / 4], rtol=1e-05, atol=1e-08
    )

    # batch size 2
    metrics_collection.reset()
    first_half = metrics_collection(scores[:2], labels[:2])
    second_half = metrics_collection(scores[2:], labels[2:])
    total = metrics_collection.compute()
    assert np.allclose(
        [
            (first_half["accuracy@1"].item() + second_half["accuracy@1"].item()) / 2,
            (first_half["accuracy@2"].item() + second_half["accuracy@2"].item()) / 2,
        ],
        [0.5 / 4, (1 + 2 / 3) / 4],
        rtol=1e-05,
        atol=1e-08,
    )
    assert np.allclose(
        [total["accuracy@1"].item(), total["accuracy@2"].item()], [0.5 / 4, (1 + 2 / 3) / 4], rtol=1e-05, atol=1e-08
    )

    # batch size 1
    metrics_collection.reset()
    results = []
    for i in range(4):
        results.append(metrics_collection(scores[i], labels[i]))
    total = metrics_collection.compute()
    mean_acc1 = np.mean([res["accuracy@1"].item() for res in results])
    mean_acc2 = np.mean([res["accuracy@2"].item() for res in results])
    assert np.allclose([mean_acc1, mean_acc2], [0.5 / 4, (1 + 2 / 3) / 4], rtol=1e-05, atol=1e-08)
    assert np.allclose(
        [total["accuracy@1"].item(), total["accuracy@2"].item()], [0.5 / 4, (1 + 2 / 3) / 4], rtol=1e-05, atol=1e-08
    )


def test_gpt2_shift(metrics_collection, metrics_collection_no_shift):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    input = "My name is John"
    tokenized_input = tokenizer(input, add_special_tokens=True, padding=False, return_tensors="pt").input_ids
    logits = model(input_ids=tokenized_input, labels=tokenized_input).logits
    preds_tokens = tokenizer.convert_ids_to_tokens(torch.topk(logits, 1, dim=-1)[1].squeeze(-1).squeeze(0))
    input_tokens = tokenizer.convert_ids_to_tokens(tokenized_input.squeeze(0))

    assert input_tokens[1:][-1] == preds_tokens[:-1][-1]

    results = metrics_collection(logits, tokenized_input)
    results_no_shift = metrics_collection_no_shift(logits, tokenized_input)

    for key in results:
        assert results[key] > results_no_shift[key]


def test_t5_shift(metrics_collection, metrics_collection_no_shift):
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    input = "Q: Can Geoffrey Hinton have a conversation with George Washington? Give the rationale before answering."
    target = "George Washington is a fictional character. Geoffrey Hinton is a fictional character."
    tokenized_input = tokenizer(input, add_special_tokens=True, padding=False, return_tensors="pt").input_ids
    tokenized_target = tokenizer(target, add_special_tokens=True, padding=False, return_tensors="pt").input_ids

    logits = model(input_ids=tokenized_input, labels=tokenized_target).logits
    preds_tokens = tokenizer.convert_ids_to_tokens(torch.topk(logits, 1, dim=-1)[1].squeeze(-1).squeeze(0))
    target_tokens = tokenizer.convert_ids_to_tokens(tokenized_target.squeeze(0))

    assert preds_tokens[:-1] == target_tokens[:-1]

    results = metrics_collection(logits, tokenized_target)
    results_no_shift = metrics_collection_no_shift(logits, tokenized_target)

    for key in results:
        assert results[key] < results_no_shift[key]


def test_full_state_update():
    scores = torch.tensor(
        [
            [[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3], [-1, -1, -1]],
            [[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3], [-1, -1, -1]],
            [[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3], [-1, -1, -1]],
            [[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3], [-1, -1, -1]],
        ],
        dtype=torch.float,
    )
    labels = torch.tensor([[-1, 0, 2, -100], [-1, 1, 1, -100], [-1, 2, 1, 3], [-1, 1, 2, -100]], dtype=torch.long)

    for top_k in list(range(1, 4)):
        check_forward_full_state_property(
            Accuracy,
            init_args=dict(top_k=top_k, ignore_index=-100),
            input_args={"predictions": scores, "references": labels},
        )
