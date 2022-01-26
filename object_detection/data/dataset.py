import math
import numpy as np
import cv2
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
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(CenternetDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(self.annotation_lines)
        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0] / 4), int(input_shape[1] / 4))
        self.num_classes = num_classes
        self.train = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        image, box = self.get_data(self.annotation_lines[index], self.input_shape)
        batch_hm = np.zeros(
            (self.output_shape[0], self.output_shape[1], self.num_classes),
            dtype=np.float32,
        )
        batch_wh = np.zeros(
            (self.output_shape[0], self.output_shape[1], 2), dtype=np.float32
        )
        batch_reg = np.zeros(
            (self.output_shape[0], self.output_shape[1], 2), dtype=np.float32
        )
        batch_reg_mask = np.zeros(
            (self.output_shape[0], self.output_shape[1]), dtype=np.float32
        )

        if len(box) != 0:
            boxes = np.array(box[:, :4], dtype=np.float32)
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

        for i in range(len(box)):
            bbox = boxes[i].copy()
            cls_id = int(box[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32
                )
                ct_int = ct.astype(np.int32)
                batch_hm[:, :, cls_id] = draw_gaussian(
                    batch_hm[:, :, cls_id], ct_int, radius
                )
                batch_wh[ct_int[1], ct_int[0]] = 1.0 * w, 1.0 * h
                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                batch_reg_mask[ct_int[1], ct_int[0]] = 1

        image = np.transpose(preprocess_input(image), (2, 0, 1))

        return image, batch_hm, batch_wh, batch_reg, batch_reg_mask

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_data(
        self, annotation_line, input_shape,
    ):
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]])

        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new("RGB", (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.float32)

        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

        return image_data, box
