import torch
from torchmetrics import Metric


class Accuracy(Metric):
    """Accuracy@k metric. Returns a ratio of examples where reference is present among top k predictions."""

    def __init__(self, top_k: int = 5, ignore_index: int = -100, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.top_k = top_k
        self.ignore_index = ignore_index

        self.add_state("accuracy", default=torch.tensor(0).float(), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, predictions: torch.Tensor, references: torch.Tensor):
        assert predictions.ndimension() == references.ndimension() + 1
        assert predictions.size()[:-1] == references.size()
        assert predictions.size()[-1] >= self.top_k

        # for support of batches of size 1
        if len(references.shape) == 1:
            references = references.unsqueeze(0)
            predictions = predictions.unsqueeze(0)

        # shift scores and labels
        predictions = predictions[..., :-1, :]
        references = references[..., 1:]

        # labels =  [batch_size x seq_len - 1]
        # scores = [batch_size x seq_len - 1 x vocab_size]
        # top_k_predictions = [batch_size x seq_len -1 x top_k]
        _, top_k_predictions = torch.topk(predictions, self.top_k)
        expanded_labels = references.unsqueeze(-1).expand_as(top_k_predictions)
        true_pos = torch.logical_and(expanded_labels == top_k_predictions, expanded_labels != self.ignore_index)

        acc_top_k_list = (
            true_pos.sum(dim=-1).float() / (references != self.ignore_index).sum(dim=1).unsqueeze(1).float()
        ).sum(dim=1)

        self.accuracy += acc_top_k_list.sum()
        self.total += references.shape[0]

    def compute(self):
        return self.accuracy.float() / self.total
