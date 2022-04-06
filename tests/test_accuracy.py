import pytest
import torch
import numpy as np
from torchmetrics import MetricCollection
from src.utils.accuracy import Accuracy


@pytest.fixture
def metrics_collection():
    return MetricCollection(
        {
            "accuracy@1": Accuracy(top_k=1, ignore_index=-100),
            "accuracy@2": Accuracy(top_k=2, ignore_index=-100),
            "accuracy@3": Accuracy(top_k=3, ignore_index=-100),
        }
    )


def test_full_top1_match(metrics_collection):
    scores = torch.tensor(
        [[0.5, 0, 0.3, 0.2, 0.4], [1.0, -100, -200, -100, 0], [-1, -1, -1, -1, -1]], dtype=torch.float
    )
    labels = torch.tensor([-1, 0, 0], dtype=torch.long)
    results = metrics_collection(scores, labels)
    assert np.allclose([results["accuracy@1"].item(), results["accuracy@3"].item()], [1.0, 1.0], rtol=1e-05, atol=1e-08)


def test_full_top3_match(metrics_collection):
    scores = torch.tensor(
        [[0.5, 0, 0.3, 0.2, 0.4], [1.0, -100, -200, -100, 0], [-1, -1, -1, -1, -1]], dtype=torch.float
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
