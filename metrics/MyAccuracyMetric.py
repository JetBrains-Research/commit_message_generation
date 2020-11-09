from pytorch_lightning.metrics import Metric
import torch


class MyAccuracyMetric(Metric):
    def __init__(self, pad_token_id, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.pad_token_id = pad_token_id

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        assert preds.shape == targets.shape

        # TODO: should we compute accuracy with token indices or is it better to compare decoded strings like in bleu?
        # TODO: should we count <s> and </s> tokens or not?
        # correct --- # of equal tokens in pred and target without padding
        self.correct += torch.sum(torch.logical_and(targets != self.pad_token_id, targets == preds))
        # total --- # of tokens in target without padding
        self.total += torch.sum(targets != self.pad_token_id)

    def compute(self):
        return self.correct.float() / self.total
