import math
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def preprocess_input(image):
    image = np.array(image, dtype=np.float32)[:, :, ::-1]
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return (image / 255.0 - mean) / std


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert("RGB")
        return image


class CenternetDataset(Dataset):
    def __init__(self, data_path, phase, num_classes, input_shape, stride, transforms):
        super(CenternetDataset, self).__init__()
        suffix = "/train.txt" if phase == "train" else "/validation.txt"
        self.data_path = data_path + suffix
        with open(self.data_path) as file:
            self.annotation_lines = file.readlines()
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0] / stride), int(input_shape[1] / stride))
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        image, labels = self.get_data(self.annotation_lines[index], self.input_shape)
        target_cls = np.zeros(
            (self.output_shape[0], self.output_shape[1], self.num_classes),
            dtype=np.float32,
        )
        target_size = np.zeros(
            (self.output_shape[0], self.output_shape[1], 2), dtype=np.float32
        )
        target_offset = np.zeros(
            (self.output_shape[0], self.output_shape[1], 2), dtype=np.float32
        )
        target_regression_mask = np.zeros(
            (self.output_shape[0], self.output_shape[1]), dtype=np.float32
        )

        if len(labels) != 0:
            boxes = np.array(labels[:, :4], dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(
                boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1],
                0,
                self.output_shape[1] - 1,
            )
            boxes[:, [1, 3]] = np.clip(
                boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0],
                0,
                self.output_shape[0] - 1,
            )

        for i in range(len(labels)):
            box = boxes[i].copy()
            cls_id = int(labels[i, -1])

            w, h = box[2] - box[0], box[3] - box[1]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32
                )
                ct_int = ct.astype(np.int32)
                target_cls[:, :, cls_id] = draw_gaussian(
                    target_cls[:, :, cls_id], ct_int, radius
                )
                target_size[ct_int[1], ct_int[0]] = 1.0 * w, 1.0 * h
                target_offset[ct_int[1], ct_int[0]] = ct - ct_int
                target_regression_mask[ct_int[1], ct_int[0]] = 1

        image = np.transpose(preprocess_input(image), (2, 0, 1))

        return image, target_cls, target_size, target_offset, target_regression_mask

    def get_data(self, annotation_line, input_shape):
        image = Image.open(annotation_line)
        image = cvtColor(image)
        iw, ih = image.size
        w, h = input_shape

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)

        with open(annotation_line.replace("images", "labels")) as file:
            labels = file.readlines()
        labels_items = []
        for label in labels:
            label_items = label.split(" ")
            label_items.append(label_items.pop(0))
            label_items[0] *= int(w)
            label_items[1] *= int(h)
            label_items[2] *= int(w)
            label_items[2] += label_items[0]
            label_items[3] *= int(h)
            label_items[3] += label_items[1]
            labels_items.append(label_items)
        labels = np.array([np.array(label_items) for label_items in labels_items])

        if len(labels) > 0:
            labels[:, [0, 2]] = labels[:, [0, 2]] * nw / iw + dx
            labels[:, [1, 3]] = labels[:, [1, 3]] * nh / ih + dy
            labels[:, 0:2][labels[:, 0:2] < 0] = 0
            labels[:, 2][labels[:, 2] > w] = w
            labels[:, 3][labels[:, 3] > h] = h
            box_w = labels[:, 2] - labels[:, 0]
            box_h = labels[:, 3] - labels[:, 1]
            labels = labels[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, labels
