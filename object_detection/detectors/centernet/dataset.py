import math
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms
from dataset import YOLODataset


def gaussian_radius(det_size, min_overlap=0.7):
    width, height = det_size

    a1 = 1
    b1 = width + height
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (width + height)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (width + height)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    x, y = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    width, height = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[x - left : x + right, y - top : y + bottom]
    masked_gaussian = gaussian[
        radius - left : radius + right, radius - top : radius + bottom
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        masked_heatmap = np.maximum(masked_heatmap, masked_gaussian * k)
        heatmap[x - left : x + right, y - top : y + bottom] = masked_heatmap
    return heatmap


class CenternetDataset(YOLODataset):
    def __init__(
        self,
        data_path,
        phase,
        num_classes,
        input_shape,
        augmentations=None,
        mixup_prob=0.0,
        mosaic4prob=0.0,
        mosaic9prob=0.0,
    ):
        super(CenternetDataset, self).__init__(
            data_path,
            phase,
            input_shape,
            augmentations,
            mixup_prob,
            mosaic4prob,
            mosaic9prob,
        )
        stride = 4
        self.output_shape = (
            int(self.input_shape[0] / stride),
            int(self.input_shape[1] / stride),
        )
        self.num_classes = num_classes

    def __getitem__(self, index):
        index = index % self.length

        original_image, image, labels_count, labels = super(
            CenternetDataset, self
        ).__getitem__(index)

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
                boxes[:, [0, 2]] / self.input_shape[0] * self.output_shape[0],
                0,
                self.output_shape[0] - 1,
            )
            boxes[:, [1, 3]] = np.clip(
                boxes[:, [1, 3]] / self.input_shape[1] * self.output_shape[1],
                0,
                self.output_shape[1] - 1,
            )

        for i in range(labels_count):
            box = boxes[i].copy()
            cls_id = int(labels[i, -1])

            w, h = box[2] - box[0], box[3] - box[1]
            if w > 0 and h > 0:
                radius = gaussian_radius((math.ceil(w), math.ceil(h)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=np.float32
                )
                ct_int = ct.astype(np.int32)
                target_cls[:, :, cls_id] = draw_gaussian(
                    target_cls[:, :, cls_id], ct_int, radius
                )
                target_size[ct_int[0], ct_int[1]] = 1.0 * w, 1.0 * h
                target_offset[ct_int[0], ct_int[1]] = ct - ct_int
                target_regression_mask[ct_int[0], ct_int[1]] = 1
        target_cls = np.transpose(target_cls, (2, 0, 1))
        target_size = np.transpose(target_size, (2, 0, 1))
        target_offset = np.transpose(target_offset, (2, 0, 1))

        return (
            original_image,
            image,
            labels_count,
            labels,
            target_cls,
            target_size,
            target_offset,
            target_regression_mask,
        )


def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def decode_bbox(pred_cls, pred_size, pred_offset, confidence, device):
    pred_cls = pool_nms(pred_cls)

    b, c, output_w, output_h = pred_cls.shape
    detects = []
    for batch in range(b):
        heat_map = pred_cls[batch].permute(1, 2, 0).view([-1, c])
        pred_wh = pred_size[batch].permute(1, 2, 0).view([-1, 2])
        pred_off = pred_offset[batch].permute(1, 2, 0).view([-1, 2])

        xv, yv = torch.meshgrid(torch.arange(0, output_w), torch.arange(0, output_h))
        xv, yv = xv.flatten().float(), yv.flatten().float()
        xv = xv.to(device)
        yv = yv.to(device)

        class_conf, class_pred = torch.max(heat_map, dim=-1)
        mask = class_conf > confidence

        pred_wh_mask = pred_wh[mask]
        pred_offset_mask = pred_off[mask]
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue

        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
        half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        bboxes = torch.cat(
            [xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h],
            dim=1,
        )
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h
        detect = torch.cat(
            [
                bboxes,
                torch.unsqueeze(class_conf[mask], -1),
                torch.unsqueeze(class_pred[mask], -1).float(),
            ],
            dim=-1,
        )
        detects.append(detect)

    return detects


def centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        new_shape = np.round(image_shape * np.min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2.0 / input_shape
        scale = input_shape / new_shape

        box_xy = (box_xy - offset) * scale
        box_wh *= scale

    box_mins = box_xy - (box_wh / 2.0)
    box_maxes = box_xy + (box_wh / 2.0)
    boxes = np.concatenate(
        [
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2],
        ],
        axis=-1,
    )
    boxes *= image_shape
    return boxes


def postprocess(
    prediction, need_nms, image_shape, input_shape, letterbox_image, nms_thres=0.4
):
    output = [None for _ in range(len(prediction))]

    for i, detections in enumerate(prediction):
        if len(detections) == 0:
            continue
        unique_labels = detections[:, -1].cpu().unique()

        if detections.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            if need_nms:
                keep = nms(detections_class[:, :4], detections_class[:, 4], nms_thres)
                max_detections = detections_class[keep]

            else:
                max_detections = detections_class

            output[i] = (
                max_detections
                if output[i] is None
                else torch.cat((output[i], max_detections))
            )

        if output[i] is not None:
            output[i] = output[i].cpu().detach().numpy()
            box_xy, box_wh = (
                (output[i][:, 0:2] + output[i][:, 2:4]) / 2,
                output[i][:, 2:4] - output[i][:, 0:2],
            )
            output[i][:, :4] = centernet_correct_boxes(
                box_xy, box_wh, input_shape, image_shape, letterbox_image
            )
    return output


def postprocess_predictions(pred, image_size, device):
    outputs = decode_bbox(pred[0], pred[1], pred[2], confidence=0.3, device=device,)
    outputs = postprocess(
        outputs,
        need_nms=False,
        image_shape=image_size,
        input_shape=image_size / 4,
        letterbox_image=False,
        nms_thres=0.4,
    )
    return outputs
