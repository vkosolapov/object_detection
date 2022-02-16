import random
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


def clip_coords(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[:, 0].clamp_(0, shape[0])
        boxes[:, 1].clamp_(0, shape[1])
        boxes[:, 2].clamp_(0, shape[0])
        boxes[:, 3].clamp_(0, shape[1])
    else:
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[0])
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[1])


def augment(image, bboxes, augmentation_pipeline):
    # image = np.asarray(image)
    result = augmentation_pipeline(image=image, bboxes=bboxes)
    return result["image"], result["bboxes"]


class YOLODataset(Dataset):
    def __init__(
        self,
        data_path,
        phase,
        input_shape,
        augmentations=None,
        mosaic4prob=0.0,
        mosaic9prob=0.0,
    ):
        super(YOLODataset, self).__init__()
        self.phase = phase
        suffix = "/train.txt" if phase == "train" else "/validation.txt"
        self.data_path = data_path + suffix
        with open(self.data_path) as file:
            self.annotation_lines = file.readlines()
        self.length = len(self.annotation_lines)
        self.input_shape = (input_shape, input_shape)
        self.augmentations = augmentations
        self.mosaic4prob = mosaic4prob
        self.mosaic9prob = mosaic9prob

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        if random.random() < self.mosaic4prob:
            image, labels = self.load_mosaic_4(index)
        elif random.random() < self.mosaic9prob:
            image, labels = self.load_mosaic_9(index)
        else:
            image, labels = self.load_image(index)

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
            (0, 5 - labels.shape[1], 0, 200 - labels.shape[0]),
            "constant",
            0.0,
        )
        return (
            original_image,
            image,
            labels_count,
            labels,
        )

    def load_image(self, index):
        annotation_line = self.annotation_lines[index]
        image = Image.open(annotation_line.strip("\n"))
        image = cvtColor(image)
        image = image.resize((self.input_shape[0], self.input_shape[1]), Image.LANCZOS)

        iw, ih = image.size
        w, h = self.input_shape
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

        return np.asarray(image), labels

    def load_mosaic_4(self, index):
        labels4 = []
        s = min(self.input_shape[0], self.input_shape[1])
        mosaic_border = [-s // 2, -s // 2]
        xc, yc = (int(random.uniform(-x, 2 * s + x)) for x in mosaic_border)
        indices = [index] + random.choices(list(range(self.length)), k=3)
        random.shuffle(indices)
        for i, index in enumerate(indices):
            img, labels = self.load_image(index)
            w, h = self.input_shape
            if i == 0:
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            img4[x1a:x2a, y1a:y2a, :] = img[x1b:x2b, y1b:y2b, :]
            padw = x1a - x1b
            padh = y1a - y1b
            if labels.size:
                labels[:, [0, 2]] += padw
                labels[:, [1, 3]] += padh
                labels4.append(labels)
        labels4 = np.concatenate(labels4, 0)
        labels4[:, [0, 2]] //= 2
        labels4[:, [1, 3]] //= 2
        clip_coords(labels4, self.input_shape)
        box_w = labels4[:, 2] - labels4[:, 0]
        box_h = labels4[:, 3] - labels4[:, 1]
        labels4 = labels4[np.logical_and(box_w > 1, box_h > 1)]
        img4 = Image.fromarray(img4)
        img4 = img4.resize((self.input_shape[0], self.input_shape[1]), Image.LANCZOS)
        img4 = np.asarray(img4)
        return img4, labels4

    def load_mosaic_9(self, index):
        labels9 = []
        s = min(self.input_shape[0], self.input_shape[1])
        mosaic_border = [-s // 3, -s // 3]
        xc, yc = (int(random.uniform(0, s)) for _ in mosaic_border)
        indices = [index] + random.choices(list(range(self.length)), k=8)
        random.shuffle(indices)
        for i, index in enumerate(indices):
            img, labels = self.load_image(index)
            w, h = self.input_shape
            if i == 0:
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)
                w0, h0 = w, h
                c = s, s, s + w, s + h
            elif i == 1:
                c = s, s - h, s + w, s
            elif i == 2:
                c = s + wp, s - h, s + wp + w, s
            elif i == 3:
                c = s + w0, s, s + w0 + w, s + h
            elif i == 4:
                c = s + w0, s + hp, s + w0 + w, s + hp + h
            elif i == 5:
                c = s + w0 - w, s + h0, s + w0, s + h0 + h
            elif i == 6:
                c = s + w0 - wp - w, s + h0, s + w0 - wp, s + h0 + h
            elif i == 7:
                c = s - w, s + h0 - h, s, s + h0
            elif i == 8:
                c = s - w, s + h0 - hp - h, s, s + h0 - hp
            padw, padh = c[:2]
            x1, y1, x2, y2 = (max(x, 0) for x in c)
            if labels.size:
                labels[:, [0, 2]] += padw
                labels[:, [1, 3]] += padh
                labels9.append(labels)
            img9[y1:y2, x1:x2, :] = img[x1 - padw :, y1 - padh :, :]
            wp, hp = w, h
        img9 = img9[xc : xc + 2 * s, yc : yc + 2 * s]
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [0, 2]] -= xc
        labels9[:, [1, 3]] -= yc
        labels9[:, [0, 2]] //= 3
        labels9[:, [1, 3]] //= 3
        clip_coords(labels9, self.input_shape)
        box_w = labels9[:, 2] - labels9[:, 0]
        box_h = labels9[:, 3] - labels9[:, 1]
        labels9 = labels9[np.logical_and(box_w > 1, box_h > 1)]
        img9 = Image.fromarray(img9)
        img9 = img9.resize((self.input_shape[0], self.input_shape[1]), Image.LANCZOS)
        img9 = np.asarray(img9)
        return img9, labels9
