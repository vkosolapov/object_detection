import torch


def compute_losses(pred, labels, criterion, losses):
    losses["cls"] = criterion["cls"](pred[0], labels[0])
    if criterion["box"] is None:
        losses["size"] = criterion["size"](pred[1], labels[1], labels[3])
        losses["offset"] = criterion["offset"](pred[2], labels[2], labels[3])
    else:
        losses["box"] = criterion["box"](
            torch.cat([pred[2], pred[1]], dim=1),
            torch.cat([labels[2], labels[1]], dim=1),
            labels[3],
        )
