import numpy as np
import torch


def augment(img, augmentation_pipeline):
    img = np.asarray(img)
    return augmentation_pipeline(image=img)["image"]


class DataLoader:
    def __init__(self, dataset, phase, batch_size, workers):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(phase == "train"),
            drop_last=(phase == "train"),
            num_workers=workers,
            persistent_workers=True,
            pin_memory=True,
        )
        self.data_loader = data_loader
        self.dataset_size = len(dataset)
