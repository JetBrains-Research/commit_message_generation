from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional.nlp import bleu_score
import torch


class MyBleuMetric(Metric):
    def __init__(self, n_gram=4, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_gram = n_gram
        self.add_state("bleu", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        self.bleu += bleu_score(translate_corpus=preds, reference_corpus=targets, n_gram=self.n_gram)
        self.total += 1

    def compute(self):
        return self.bleu / self.total
