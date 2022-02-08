def compute_losses(pred, labels, criterion, losses):
    losses["cls"] = criterion["cls"](pred[0], labels[0])
    losses["size"] = criterion["size"](pred[1], labels[1], labels[3])
    losses["offset"] = criterion["offset"](pred[2], labels[2], labels[3])
