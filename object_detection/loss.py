import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        pred = pred.permute(0, 2, 3, 1)
        expand_mask = torch.unsqueeze(mask,-1).repeat(1, 1, 1, 2)
        loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class LabelSmoothingFocalLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        gamma: int = 0,
        alpha: float = None,
        smoothing: float = 0.0,
        size_average: bool = True,
        ignore_index: int = None,
    ):
        super().__init__()
        self._num_classes = num_classes
        self._gamma = gamma
        self._alpha = alpha
        self._smoothing = smoothing
        self._size_average = size_average
        self._ignore_index = ignore_index
        self._log_softmax = nn.LogSoftmax(dim=1)

        if self._num_classes <= 1:
            raise ValueError("The number of classes must be 2 or higher")
        if self._gamma < 0:
            raise ValueError("Gamma must be 0 or higher")
        if self._alpha is not None:
            if self._alpha <= 0 or self._alpha >= 1:
                raise ValueError("Alpha must be 0 <= alpha <= 1")

    def forward(self, logits, label):
        logits = logits.float()
        difficulty_level = self._estimate_difficulty_level(logits, label)

        with torch.no_grad():
            label = label.clone().detach()
            if self._ignore_index is not None:
                ignore = label.eq(self._ignore_index)
                label[ignore] = 0
            lb_pos, lb_neg = (
                1.0 - self._smoothing,
                self._smoothing / (self._num_classes - 1),
            )
            lb_one_hot = (
                torch.empty_like(logits)
                .fill_(lb_neg)
                .scatter_(1, label.unsqueeze(1), lb_pos)
                .detach()
            )
        logs = self._log_softmax(logits)
        loss = -torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
        if self._ignore_index is not None:
            loss[ignore] = 0
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):
        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level
