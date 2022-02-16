import numpy as np
import torch


class DataLoader:
    def __init__(self, dataset, phase, batch_size, workers, device):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(phase == "train"),
            drop_last=(phase == "train"),
            num_workers=workers,
            persistent_workers=True,
            pin_memory=(device != "cpu"),
        )
        self.data_loader = data_loader
        self.dataset_size = len(dataset)
