from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.nlp import bleu_score
import torch


class AccuracyMetric(Metric):
    def __init__(self, pad_token_id, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.pad_token_id = pad_token_id

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape

        # correct --- # of equal tokens in pred and target without <s> </s> <pad>
        self.correct += torch.sum(torch.logical_and(targets != self.pad_token_id, targets == preds))

        # total --- # of tokens in target without <s> </s> <pad>
        self.total += torch.sum(targets != self.pad_token_id)

    def compute(self):
        return self.correct.float() / self.total


class BleuMetric(Metric):
    def __init__(self, n_gram=4, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_gram = n_gram
        self.add_state("bleu", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        for pred, trg in zip(preds, targets):
            self.bleu += bleu_score(translate_corpus=[pred.split(' ')], reference_corpus=[[trg.split(' ')]],
                                    n_gram=self.n_gram, smooth=True)
            self.total += 1

    def compute(self):
        return self.bleu / self.total
