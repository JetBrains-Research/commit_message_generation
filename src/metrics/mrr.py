import torch
from torchmetrics import Metric


class MRR(Metric):
    """Mean Reciprocal Rank (MRR)@k metric. In contrast with accuracy, it takes a position of correct prediction among
    top k into account."""

    def __init__(self, top_k: int = 5, ignore_index: int = -100, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.top_k = top_k
        self.ignore_index = ignore_index

        self.add_state("mrr", default=torch.tensor(0).float(), dist_reduce_fx="sum")
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
        # top_k_predictions = [batch_size x seq_len - 1 x top_k]
        _, top_k_predictions = torch.topk(predictions, self.top_k)
        expanded_labels = references.unsqueeze(-1).expand_as(top_k_predictions)
        true_pos = torch.logical_and(expanded_labels == top_k_predictions, expanded_labels != self.ignore_index)
        # mrr depends on position of correct label in top k generated outputs
        true_pos_for_mrr = true_pos / torch.arange(1, true_pos.size(-1) + 1, dtype=torch.float, device=true_pos.device)
        mrr_top_k_list = (
            true_pos_for_mrr.max(dim=-1)[0].sum(dim=-1) / (references != self.ignore_index).sum(dim=1).float()
        )

        try:
            self.mrr += mrr_top_k_list.sum()
            self.total += references.shape[0]
        except RuntimeError:
            self.mrr = self.mrr.to(mrr_top_k_list.device)
            self.total = self.total.to(self.mrr.device)

    def compute(self):
        return self.mrr.float() / self.total
