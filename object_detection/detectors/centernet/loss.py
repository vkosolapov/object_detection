import torch


def offset_to_coord(pred):
    b, c, output_w, output_h = pred.shape
    xv, yv = torch.meshgrid(torch.arange(0, output_w), torch.arange(0, output_h))
    coords = torch.stack([xv, yv]).repeat(b, 1, 1, 1)
    return pred + coords


def compute_losses(pred, labels, criterion, losses):
    losses["cls"] = criterion["cls"](pred[0], labels[0])
    if criterion["box"] is None:
        losses["size"] = criterion["size"](pred[1], labels[1], labels[3])
        losses["offset"] = criterion["offset"](pred[2], labels[2], labels[3])
    else:
        pred_coord = offset_to_coord(pred[2])
        labels_coord = offset_to_coord(labels[2])
        losses["box"] = criterion["box"](
            torch.cat([pred_coord, pred[1]], dim=1),
            torch.cat([labels_coord, labels[1]], dim=1),
            labels[3],
        )
