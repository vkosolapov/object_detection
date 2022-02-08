import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        pred = pred.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        expand_mask = torch.unsqueeze(mask, -1).repeat(1, 1, 1, 2)
        loss = F.smooth_l1_loss(
            pred * expand_mask, target * expand_mask, reduction="sum"
        )
        loss = loss / (expand_mask.sum() + 1e-4)
        return loss


def compute_losses(pred, labels, criterion, losses):
    losses["cls"] = criterion["cls"](pred[0], labels[0])
    losses["size"] = criterion["size"](pred[1], labels[1], labels[3])
    losses["offset"] = criterion["offset"](pred[2], labels[2], labels[3])
