import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert("RGB")
        return image


def preprocess_input(image):
    image = np.array(image, dtype=np.float32)[:, :, ::-1]
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    image = (image / 255.0 - mean) / std
    return torch.Tensor(image).type(dtype=torch.float32).permute(2, 0, 1)


def augment(image, bboxes, augmentation_pipeline):
    image = np.asarray(image)
    result = augmentation_pipeline(image=image, bboxes=bboxes)
    return result["image"], result["bboxes"]


class YOLODataset(Dataset):
    def __init__(self, data_path, phase, input_shape, augmentations=None):
        super(YOLODataset, self).__init__()
        self.phase = phase
        suffix = "/train.txt" if phase == "train" else "/validation.txt"
        self.data_path = data_path + suffix
        with open(self.data_path) as file:
            self.annotation_lines = file.readlines()
        self.length = len(self.annotation_lines)
        self.input_shape = (input_shape, input_shape)
        self.augmentations = augmentations

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        image, labels = self.get_data(index, self.input_shape)

        if self.phase == "train":
            image, labels = augment(image, labels, self.augmentations)
            labels = np.array(labels, dtype=np.int32)
        else:
            image = np.array(image, np.float32)
        original_image = image.copy()
        image = preprocess_input(image)

        if len(labels.shape) < 2:
            labels = np.expand_dims(labels, axis=1)
        labels_count = labels.shape[0]
        labels = F.pad(
            torch.Tensor(labels),
            (0, 5 - labels.shape[1], 0, 100 - labels.shape[0]),
            "constant",
            0.0,
        )
        return (
            original_image,
            image,
            labels_count,
            labels,
        )

    def get_data(self, index, input_shape):
        annotation_line = self.annotation_lines[index]
        image = Image.open(annotation_line.strip("\n"))
        image = cvtColor(image)
        image = image.resize((self.input_shape[0], self.input_shape[1]), Image.LANCZOS)

        iw, ih = image.size
        w, h = input_shape
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        with open(
            annotation_line.replace("images", "labels")
            .replace("jpg", "txt")
            .strip("\n")
        ) as file:
            labels = file.readlines()
        labels_items = []
        for label in labels:
            label_items = label.strip("\n").split(" ")
            label_items.append(label_items.pop(0))
            label_items[0] = int(float(label_items[0]) * iw)
            label_items[1] = int(float(label_items[1]) * ih)
            label_items[2] = int(float(label_items[2]) * iw)
            label_items[2] += label_items[0]
            label_items[3] = int(float(label_items[3]) * ih)
            label_items[3] += label_items[1]
            labels_items.append(label_items)
        labels = np.array(labels_items, dtype=np.int32)

        if len(labels) > 0:
            labels[:, [0, 2]] = labels[:, [0, 2]] * nw / iw + dx
            labels[:, [1, 3]] = labels[:, [1, 3]] * nh / ih + dy
            labels[:, 0:2][labels[:, 0:2] < 0] = 0
            labels[:, 2][labels[:, 2] > w] = w
            labels[:, 3][labels[:, 3] > h] = h
            box_w = labels[:, 2] - labels[:, 0]
            box_h = labels[:, 3] - labels[:, 1]
            labels = labels[np.logical_and(box_w > 1, box_h > 1)]

        return image, labels
