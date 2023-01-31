import numpy as np
import pytest
import torch
from torchmetrics import MetricCollection
from torchmetrics.utilities import check_forward_full_state_property

from src.metrics.mrr import MRR


@pytest.fixture
def metrics_collection():
    return MetricCollection(
        {
            "MRR@1": MRR(top_k=1, ignore_index=-100),
            "MRR@2": MRR(top_k=2, ignore_index=-100),
            "MRR@3": MRR(top_k=3, ignore_index=-100),
        }
    )


def test_full_top1_match(metrics_collection):
    scores = torch.tensor(
        [[0.5, 0, 0.3, 0.2, 0.4], [1.0, -100, -200, -100, 0], [-1, -1, -1, -1, -1]], dtype=torch.float
    )
    labels = torch.tensor([-1, 0, 0], dtype=torch.long)
    results = metrics_collection(scores, labels)
    assert np.allclose([results["MRR@1"].item(), results["MRR@3"].item()], [1.0, 1.0], rtol=1e-05, atol=1e-08)


def test_full_top3_match(metrics_collection):
    scores = torch.tensor(
        [[0.5, 0, 0.3, 0.2, 0.4], [1.0, -100, -200, -101, 0], [-1, -1, -1, -1, -1]], dtype=torch.float
    )
    labels = torch.tensor([-1, 2, 1], dtype=torch.long)
    results = metrics_collection(scores, labels)
    print(results["MRR@1"].item(), results["MRR@3"].item())
    assert np.allclose(
        [results["MRR@1"].item(), results["MRR@3"].item()], [0.0, (1 / 3 + 1 / 3) / 2], rtol=1e-05, atol=1e-08
    )


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
    assert np.allclose([results["MRR@2"].item()], [(0.75 + 1 / 3) / 4], rtol=1e-05, atol=1e-08)

    # batch size 2
    metrics_collection.reset()
    first_half = metrics_collection(scores[:2], labels[:2])
    second_half = metrics_collection(scores[2:], labels[2:])
    total = metrics_collection.compute()
    assert np.allclose(
        [(first_half["MRR@2"].item() + second_half["MRR@2"].item()) / 2], [(0.75 + 1 / 3) / 4], rtol=1e-05, atol=1e-08
    )
    assert np.allclose([total["MRR@2"].item()], [(0.75 + 1 / 3) / 4], rtol=1e-05, atol=1e-08)

    # batch size 1
    metrics_collection.reset()
    results = []
    for i in range(4):
        results.append(metrics_collection(scores[i], labels[i]))
    total = metrics_collection.compute()
    assert np.allclose(
        [np.mean([res["MRR@2"].item() for res in results])], [(0.75 + 1 / 3) / 4], rtol=1e-05, atol=1e-08
    )
    assert np.allclose([total["MRR@2"].item()], [(0.75 + 1 / 3) / 4], rtol=1e-05, atol=1e-08)


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
            MRR,
            init_args=dict(top_k=top_k, ignore_index=-100),
            input_args={"predictions": scores, "references": labels},
        )
