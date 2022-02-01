import numpy as np
from functools import partial
import torch
from torchvision import transforms


mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def augment(img, augmentation_pipeline):
    img = np.asarray(img)
    return augmentation_pipeline(image=img)["image"]


class DataLoader:
    def __init__(
        self, dataset, phase, batch_size, augmentations, workers,
    ):
        data_transforms = self.transforms(augmentations)
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

    def transforms(self, augmentations):
        return {
            "train": transforms.Compose(
                [
                    partial(augment, augmentation_pipeline=augmentations),
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }


def rename(file_name, string_to_add):
    with open(file_name, "r") as f:
        file_lines = ["".join([string_to_add, x.strip(), "\n"]) for x in f.readlines()]
    with open(file_name, "w") as f:
        f.writelines(file_lines)


if __name__ == "__main__":
    rename("data/AFO/PART_1/PART_1/train.txt", "data/AFO/PART_1/PART_1/images/")
    rename("data/AFO/PART_1/PART_1/validation.txt", "data/AFO/PART_1/PART_1/images/")
    rename("data/AFO/PART_1/PART_1/test.txt", "data/AFO/PART_1/PART_1/images/")
