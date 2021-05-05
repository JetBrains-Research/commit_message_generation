import unittest
import torch
import numpy as np
from metrics import accuracy_MRR


class TestMetrics(unittest.TestCase):
    def test_top1_match(self):
        scores = torch.tensor([[0.5, 0, 0.3, 0.2, 0.4], [1.0, -100, -200, -100, 0]], dtype=torch.float)
        labels = torch.tensor([0, 0], dtype=torch.long)
        acc_top1, acc_top5, MRR_top5 = accuracy_MRR(scores, labels, 5, shift=False)
        self.assertTrue(np.allclose([acc_top1, acc_top5, MRR_top5], [1.0, 1.0, 1.0], rtol=1e-05, atol=1e-08))

    def test_top3_match(self):
        scores = torch.tensor([[0.5, 0, 0.3, 0.2, 0.4], [1.0, -100, -200, -100, 0]], dtype=torch.float)
        labels = torch.tensor([0, 2], dtype=torch.long)
        acc_top1, acc_top3, MRR_top3 = accuracy_MRR(scores, labels, 3, shift=False)
        self.assertTrue(np.allclose([acc_top1, acc_top3, MRR_top3], [0.5, 0.5, 0.5], rtol=1e-05, atol=1e-08))

    def test_top5_match(self):
        scores = torch.tensor([[0.5, 0, 0.3, 0.2, 0.4], [1.0, -100, -200, -100, 0]], dtype=torch.float)
        labels = torch.tensor([1, 2], dtype=torch.long)
        acc_top1, acc_top5, MRR_top5 = accuracy_MRR(scores, labels, 5, shift=False)
        self.assertTrue(np.allclose([acc_top1, acc_top5, MRR_top5], [0.0, 1.0, 0.2], rtol=1e-05, atol=1e-08))

    def test_different_batch_sizes(self):
        # batch size 4
        scores = torch.tensor([[[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3]],
                               [[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3]],
                               [[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3]],
                               [[0.5, 0, 0.3], [1.0, -100, -200], [0.5, 0, 0.3]]], dtype=torch.float)
        labels = torch.tensor([[0, 2, -100], [1, 1, -100], [2, 1, 3], [1, 2, -100]],
                              dtype=torch.long)
        acc_top1, acc_top2, MRR_top2 = accuracy_MRR(scores, labels, 2, shift=False)
        self.assertTrue(
            np.allclose([acc_top1, acc_top2, MRR_top2], [0.5 / 4, (1 + 2 / 3) / 4, (0.75 + 1 / 3) / 4], rtol=1e-05,
                        atol=1e-08))

        # batch size 2
        first_half = np.array(accuracy_MRR(scores[:2], labels[:2], 2, shift=False))
        second_half = np.array(accuracy_MRR(scores[2:], labels[2:], 2, shift=False))
        res = (first_half + second_half) / 2
        self.assertTrue(
            np.allclose(res, [0.5 / 4, (1 + 2 / 3) / 4, (0.75 + 1 / 3) / 4], rtol=1e-05,
                        atol=1e-08))

        # batch size 1
        results = []
        for i in range(4):
            results.append(np.array(accuracy_MRR(scores[i], labels[i], 2, shift=False)))
        self.assertTrue(
            np.allclose(np.mean(results, axis=0), [0.5 / 4, (1 + 2 / 3) / 4, (0.75 + 1 / 3) / 4], rtol=1e-05,
                        atol=1e-08))

    def test_with_shift(self):
        # Will ignore last. So actual scores are [[0.5, 0, 0.3, 0.2, 0.4], [1.0, -100, -200, -101, 0]]
        scores = torch.tensor([[0.5, 0, 0.3, 0.2, 0.4], [1.0, -100, -200, -101, 0], [1, 2, 0, 4, 5]], dtype=torch.float)
        # Will ignore first. So actual labels are [0, 1]
        labels = torch.tensor([0, 0, 1], dtype=torch.long)
        acc_top1, acc_top3, MRR_top3 = accuracy_MRR(scores, labels, 3, ignore_index=-100, shift=True)
        self.assertTrue(np.allclose([acc_top1, acc_top3, MRR_top3], [0.5, 1.0, (1 + 1/3)/2], rtol=0, atol=1e-06))


if __name__ == '__main__':
    unittest.main()
