from typing import Tuple
import torch


def accuracy_MRR(
    scores: torch.Tensor,
    labels: torch.Tensor,
    top_k: int = 5,
    ignore_index: int = -100,
    shift: bool = False,
) -> Tuple[float, float, float]:
    """Calculates accuracy@1, accuracy@top_k and MRR@top_k given scores (softmax or logits) and ground truth labels.

    :param scores: logits or softmax scores
    :param labels: ground truth labels
    :param top_k: parameter of calculated accuracy@top_k and MRR@top_k
    :param ignore_index: index that should be ignored when computing metrics. If you don't need it, just pass negative
    :param shift: whether your model works with sequences and inputs == target. It's always true for HuggingFace models
    :return: tuple of 3 floats: (accuracy@1, accuracy@top_k, MRR@top_k)
    """
    assert scores.ndimension() == labels.ndimension() + 1
    assert scores.size()[:-1] == labels.size()
    assert scores.size()[-1] >= top_k

    if shift:
        scores = scores[..., :-1, :]
        labels = labels[..., 1:]

    if len(labels.shape) == 1:
        labels = labels.unsqueeze(0)
        scores = scores.unsqueeze(0)

    # labels =  [batch_size x seq_len - 1]
    # scores = [batch_size x seq_len - 1 x vocab_size]
    # true_pos = [batch_size x seq_len -1 x top_k]
    _, top_k_predictions = torch.topk(scores, top_k)
    expanded_labels = labels.unsqueeze(-1).expand_as(top_k_predictions)
    true_pos = torch.logical_and(expanded_labels == top_k_predictions, expanded_labels != ignore_index)

    acc_top_1_list = (true_pos[..., :1].sum(dim=1).flatten() /
                      (labels != ignore_index).sum(dim=1).float())
    acc_top_1 = torch.mean(acc_top_1_list).item()

    acc_top_k_list = (true_pos.sum(dim=-1).float() /
                                (labels != ignore_index).sum(dim=1).unsqueeze(1).float()).sum(dim=1)

    acc_top_k = torch.mean(acc_top_k_list).item()

    true_pos_for_MRR = true_pos / torch.arange(1, true_pos.size(-1) + 1, dtype=torch.float, device=true_pos.device)
    MRR_top_k_list = true_pos_for_MRR.max(dim=-1)[0].sum(dim=-1) / (labels != ignore_index).sum(dim=1).float()
    MRR_top_k = torch.mean(MRR_top_k_list).item()

    return acc_top_1, acc_top_k, MRR_top_k