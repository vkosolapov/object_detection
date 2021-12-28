import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt


IMAGE_SIZE = 800
SUB_SAMPLE = 16
FEATURE_SIZE = IMAGE_SIZE // SUB_SAMPLE

RATIOS = [0.5, 1.0, 2.0]
SCALES = [8, 16, 32]
K = len(RATIOS) * len(SCALES)
ANCHORS_COUNT = FEATURE_SIZE * FEATURE_SIZE
ANCHOR_BOXES_COUNT = ANCHORS_COUNT * K

RPN_POSITIVE_IOU_THRESHOLD = 0.7
RPN_NEGATIVE_IOU_THRESHOLD = 0.3
RPN_POSITIVE_RATIO = 0.5
RPN_SAMPLES = 256
RPN_POSITIVE_COUNT = int(RPN_SAMPLES * RPN_POSITIVE_RATIO)
RPN_INPUT_CHANNELS = 512
RPN_HIDDEN_CHANNELS = 512
RPN_LAMBDA = 10.0

NMS_THRESHOLD = 0.7
N_TRAIN_PRE_NMS = 12000
N_TRAIN_POST_NMS = 2000
N_TEST_PRE_NMS = 6000
N_TEST_POST_NMS = 300
MIN_SIZE = 16

RCNN_SAMPLES = 128
RCNN_POSITIVE_RATIO = 0.25
RCNN_POSITIVE_COUNT = int(RCNN_SAMPLES * RCNN_POSITIVE_RATIO)
RCNN_POSITIVE_IOU_THRESHOLD = 0.5
RCNN_NEGATIVE_IOU_THRESHOLD_HI = 0.5
RCNN_NEGATIVE_IOU_THRESHOLD_LO = 0.0
RCNN_LAMBDA = 10.0

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def show_boxes_and_labels(image, bboxes, labels):
    image_clone = np.copy(image)
    for i in range(len(bboxes)):
        cv2.rectangle(
            image_clone,
            (bboxes[i][1], bboxes[i][0]),
            (bboxes[i][3], bboxes[i][2]),
            color=(0, 255, 0),
            thickness=3)
        cv2.putText(
            image_clone,
            str(int(labels[i])),
            (bboxes[i][3], bboxes[i][2]),
            cv2.FONT_HERSHEY_COMPLEX,
            fontScale=3,
            color=(0, 0, 255),
            thickness=3
        )
    plt.imshow(image_clone)
    plt.show()


def preprocess(image, bboxes, labels):
    scaled_image = cv2.resize(image, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    h_ratio = IMAGE_SIZE / image.shape[0]
    w_ratio = IMAGE_SIZE / image.shape[1]
    ratio_list = [h_ratio, w_ratio, h_ratio, w_ratio]
    scaled_bboxes = []
    for bbox in bboxes:
        bbox = [int(a * b) for a, b in zip(bbox, ratio_list)]
        scaled_bboxes.append(bbox)
    scaled_bboxes = np.array(scaled_bboxes)
    return scaled_image, scaled_bboxes, labels


img0 = cv2.imread("example.jpg")
img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
bbox0 = np.array([[160, 147, 260, 234], [139, 312, 200, 348]])
labels0 = np.array([1, 1])
show_boxes_and_labels(img0, bbox0, labels0)
img, bbox, labels = preprocess(img0, bbox0, labels0)
show_boxes_and_labels(img, bbox, labels)


def get_feature_extractor(device):
    model = torchvision.models.vgg16(pretrained=True).to(device)
    features = list(model.features)
    dummy_image = torch.zeros((1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])).float()
    dummy_image = dummy_image.to(device)
    required_features = []
    for feature in features:
        dummy_image = feature(dummy_image)
        if dummy_image.size()[2] < FEATURE_SIZE:
            break
        required_features.append(feature)
    extractor = nn.Sequential(*required_features)
    return extractor


feature_extractor = get_feature_extractor(device)
transform = transforms.Compose([transforms.ToTensor()])
img_tensor = transform(img).to(device).unsqueeze(0)
out_map = feature_extractor(img_tensor)


def generate_anchors():
    anchors_x = np.arange(SUB_SAMPLE, (FEATURE_SIZE + 1) * SUB_SAMPLE, SUB_SAMPLE)
    anchors_y = np.arange(SUB_SAMPLE, (FEATURE_SIZE + 1) * SUB_SAMPLE, SUB_SAMPLE)
    anchors = np.zeros((ANCHORS_COUNT, 2))
    index = 0
    for x in range(len(anchors_x)):
        for y in range(len(anchors_y)):
            anchors[index, 1] = anchors_x[x] - SUB_SAMPLE / 2.
            anchors[index, 0] = anchors_y[y] - SUB_SAMPLE / 2.
            index += 1
    return anchors


def generate_anchor_boxes(anchors):
    anchor_boxes = np.zeros((ANCHOR_BOXES_COUNT, 4))
    index = 0
    for anchor in anchors:
        anchor_y, anchor_x = anchor
        for i in range(len(RATIOS)):
            for j in range(len(SCALES)):
                h = SUB_SAMPLE * SCALES[j] * np.sqrt(RATIOS[i])
                w = SUB_SAMPLE * SCALES[j] * np.sqrt(1 / RATIOS[i])
                anchor_boxes[index, 0] = anchor_y - h / 2.
                anchor_boxes[index, 1] = anchor_x - w / 2.
                anchor_boxes[index, 2] = anchor_y + h / 2.
                anchor_boxes[index, 3] = anchor_x + w / 2.
                index += 1
    index_inside = np.where(
        (anchor_boxes[:, 0] >= 0) &
        (anchor_boxes[:, 1] >= 0) &
        (anchor_boxes[:, 2] <= IMAGE_SIZE) &
        (anchor_boxes[:, 3] <= IMAGE_SIZE)
    )[0]
    valid_anchor_boxes = anchor_boxes[index_inside]
    return anchor_boxes, index_inside, valid_anchor_boxes


anchors = generate_anchors()
anchor_boxes, index_inside, valid_anchor_boxes = generate_anchor_boxes(anchors)


def calc_ious(bbox, valid_anchor_boxes):
    ious = np.empty((len(valid_anchor_boxes), 2), dtype=np.float32)
    ious.fill(0)
    for num1, i in enumerate(valid_anchor_boxes):
        ya1, xa1, ya2, xa2 = i
        anchor_area = (ya2 - ya1) * (xa2 - xa1)
        for num2, j in enumerate(bbox):
            yb1, xb1, yb2, xb2 = j
            box_area = (yb2 - yb1) * (xb2 - xb1)
            inter_x1 = max([xb1, xa1])
            inter_y1 = max([yb1, ya1])
            inter_x2 = max([xb2, xa2])
            inter_y2 = max([yb2, yb1])
            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                iou = inter_area / (anchor_area + box_area - inter_area)
            else:
                iou = 0
            ious[num1, num2] = iou
    return ious


def calc_cls_target(ious, index_inside):
    gt_argmax_ious = ious.argmax(axis=0)
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
    gt_argmax_ious = np.where(ious == gt_max_ious)[0]
    argmax_ious = ious.argmax(axis=1)
    max_ious = ious[np.arange(len(index_inside)), argmax_ious]

    rpn_labels = np.empty((len(index_inside),), dtype=np.int32)
    rpn_labels.fill(-1)
    rpn_labels[gt_argmax_ious] = 1
    rpn_labels[max_ious >= RPN_POSITIVE_IOU_THRESHOLD] = 1
    rpn_labels[max_ious < RPN_NEGATIVE_IOU_THRESHOLD] = 0
    return rpn_labels, argmax_ious


def sample_cls_target(rpn_labels):
    positive_index = np.where(rpn_labels == 1)[0]
    if len(positive_index) > RPN_POSITIVE_COUNT:
        disable_index = np.random.choice(positive_index, size=(len(positive_index) - RPN_POSITIVE_COUNT), replace=False)
        rpn_labels[disable_index] = -1
    negative_count = RPN_SAMPLES * np.sum(rpn_labels == 1)
    negative_index = np.where(rpn_labels == 0)[0]
    if len(negative_index) > negative_count:
        disable_index = np.random.choice(negative_index, size=(len(negative_index) - negative_count), replace=False)
        rpn_labels[disable_index] = -1


def calc_reg_target(max_iou_bbox, valid_anchor_boxes):
    height = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
    width = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
    ctr_y = valid_anchor_boxes[:, 0] + 0.5 * height
    ctr_x = valid_anchor_boxes[:, 1] + 0.5 * width
    base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
    base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
    base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)
    anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
    return anchor_locs


def rpn_target(bbox, anchor_boxes, index_inside, valid_anchor_boxes):
    ious = calc_ious(bbox, valid_anchor_boxes)
    rpn_labels, argmax_ious = calc_cls_target(ious, index_inside)
    sample_cls_target(rpn_labels)
    max_iou_bbox = bbox[argmax_ious]
    anchor_locs = calc_reg_target(max_iou_bbox, valid_anchor_boxes)

    anchor_labels = np.empty((len(anchor_boxes),), dtype=rpn_labels.dtype)
    anchor_labels.fill(-1)
    anchor_labels[index_inside] = rpn_labels
    anchor_locations = np.empty((len(anchor_boxes),) + anchor_boxes.shape[1:], dtype=anchor_locs.dtype)
    anchor_locations.fill(0)
    anchor_locations[index_inside, :] = anchor_locs

    return anchor_labels, anchor_locations


anchor_labels, anchor_locations = rpn_target(bbox, anchor_boxes, index_inside, valid_anchor_boxes)

conv1 = nn.Conv2d(RPN_INPUT_CHANNELS, RPN_HIDDEN_CHANNELS, 3, 1, 1).to(device)
conv1.weight.data.normal_(0, 0.01)
conv1.bias.data.zero_()
reg_layer = nn.Conv2d(RPN_HIDDEN_CHANNELS, K * 4, 1, 1, 0).to(device)
reg_layer.weight.data.normal_(0, 0.01)
reg_layer.bias.data.zero_()
cls_layer = nn.Conv2d(RPN_HIDDEN_CHANNELS, K * 2, 1, 1, 0).to(device)
cls_layer.weight.data.normal_(0, 0.01)
cls_layer.bias.data.zero_()

x = conv1(out_map.to(device))
pred_anchor_locs = reg_layer(x)
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
pred_cls_scores = cls_layer(x)
pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
objectness_scores = pred_cls_scores.view(1, FEATURE_SIZE, FEATURE_SIZE, ANCHORS_COUNT, 2)[:, :, :, :, 1]
objectness_scores = objectness_scores.contiguous().view(1, -1)
pred_cls_scores = pred_cls_scores.view(1, -1, 2)


def calc_reg_loss(rpn_loc, gt_rpn_loc, gt_rpn_score):
    pos = gt_rpn_score > 0
    mask = pos.unsqueeze(1).expand_as(rpn_loc)
    mask_loc_preds = rpn_loc[mask].view(-1, 4)
    mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)
    x = torch.abs(mask_loc_targets.cpu() - mask_loc_preds.cpu())
    rpn_loc_loss = ((x < 1).float() * 0.5 * x ** 2) + ((x >= 1).float() * (x - 0.5))
    n_reg = (gt_rpn_score > 0).float().sum()
    rpn_loc_loss = rpn_loc_loss.sum() / n_reg
    return rpn_loc_loss


def calc_rpn_loss(pred_locations, pred_labels, anchor_locations, anchor_labels, device):
    rpn_loc = pred_locations[0]
    rpn_score = pred_labels[0]
    gt_rpn_loc = torch.from_numpy(anchor_locations)
    gt_rpn_score = torch.from_numpy(anchor_labels)
    rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score.long().to(device), ignore_index=-1)
    rpn_loc_loss = calc_reg_loss(rpn_loc, gt_rpn_loc, gt_rpn_score)
    loss = rpn_cls_loss + RPN_LAMBDA * rpn_loc_loss
    return loss


rpn_loss = calc_rpn_loss(pred_anchor_locs, pred_cls_scores, anchor_locations, anchor_labels, device)


def non_maximum_suppression(anchor_boxes, anchor_locations, pred_anchor_locs, objectness_scores):
    anchor_height = anchor_boxes[:, 2] - anchor_boxes[:, 0]
    anchor_width = anchor_boxes[:, 3] - anchor_boxes[:, 1]
    anchor_ctr_y = anchor_boxes[:, 0] + 0.5 * anchor_height
    anchor_ctr_x = anchor_boxes[:, 1] + 0.5 * anchor_width
    pred_anchor_locs_np = pred_anchor_locs[0].cpu().data.numpy()
    objectness_scores_np = objectness_scores[0].cpu().data.numpy()
    dy = pred_anchor_locs_np[:, 0::4]
    dx = pred_anchor_locs_np[:, 1::4]
    dh = pred_anchor_locs_np[:, 2::4]
    dw = pred_anchor_locs_np[:, 3::4]
    ctr_y = dy * anchor_height[:, np.newaxis] + anchor_ctr_y[:, np.newaxis]
    ctr_x = dx * anchor_width[:, np.newaxis] + anchor_ctr_x[:, np.newaxis]
    h = np.exp(dh) * anchor_height[:, np.newaxis]
    w = np.exp(dw) * anchor_width[:, np.newaxis]

    roi = np.zeros(pred_anchor_locs_np.shape, dtype=anchor_locations.dtype)
    roi[:, 0::4] = ctr_y - 0.5 * h
    roi[:, 1::4] = ctr_x - 0.5 * w
    roi[:, 2::4] = ctr_y + 0.5 * h
    roi[:, 3::4] = ctr_x + 0.5 * w
    roi[: slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, IMAGE_SIZE)
    roi[: slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, IMAGE_SIZE)

    hs = roi[:, 2] - roi[:, 0]
    ws = roi[:, 3] - roi[:, 1]
    keep = np.where((hs >= MIN_SIZE) & (ws >= MIN_SIZE))[0]
    roi = roi[keep, :]
    score = objectness_scores_np[keep]
    order = score.reval().argsort()[::-1]
    order = order[:N_TRAIN_PRE_NMS]
    roi = roi[order, :]

    y1 = roi[:, 0]
    x1 = roi[:, 1]
    y2 = roi[:, 2]
    x2 = roi[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = order.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.maximum(x2[i], x2[order[1:]])
        yy2 = np.maximum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESHOLD)[0]
        order = order[inds + 1]
    keep = keep[:N_TRAIN_POST_NMS]
    roi = roi[keep]
    return roi


roi = non_maximum_suppression(anchor_boxes, anchor_locations, pred_anchor_locs, objectness_scores)
ious = calc_ious(bbox, roi)

gt_assignment = ious.argmax(axis=1)
max_iou = ious.max(axis=1)
gt_roi_label = labels[gt_assignment]

pos_index = np.where(max_iou >= RCNN_POSITIVE_IOU_THRESHOLD)[0]
pos_roi_per_this_image = int(min(RCNN_POSITIVE_COUNT, pos_index.size))
if pos_index.size > 0:
    pos_index = np.random.choice(pos_index, size=pos_roi_per_this_image, replace=False)
neg_index = np.where((max_iou < RCNN_NEGATIVE_IOU_THRESHOLD_HI) & (max_iou >= RCNN_NEGATIVE_IOU_THRESHOLD_LO))[0]
neg_roi_per_this_image = RCNN_SAMPLES - pos_roi_per_this_image
neg_roi_per_this_image = int(min(neg_roi_per_this_image, neg_index.size))
if neg_index.size > 0:
    neg_index = np.random.choice(neg_index, size=neg_roi_per_this_image, replace=False)

keep_index = np.append(pos_index, neg_index)
gt_roi_labels = gt_roi_label[keep_index]
gt_roi_labels[pos_roi_per_this_image:] = 0
sample_roi = roi[keep_index]

bbox_for_sampled_roi = bbox[gt_assignment[keep_index]]
height = sample_roi[:, 2] - sample_roi[:, 0]
width = sample_roi[:, 3] - sample_roi[:, 1]
ctr_y = sample_roi[:, 0] + 0.5 * height
ctr_x = sample_roi[:, 1] + 0.5 * width
base_height = bbox_for_sampled_roi[:, 2] - bbox_for_sampled_roi[:, 0]
base_width = bbox_for_sampled_roi[:, 3] - bbox_for_sampled_roi[:, 1]
base_ctr_y = bbox_for_sampled_roi[:, 0] + 0.5 * base_height
base_ctr_x = bbox_for_sampled_roi[:, 1] + 0.5 * base_width

eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)
dy = (base_ctr_y - ctr_y) / height
dx = (base_ctr_x - ctr_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)
gt_roi_locs = np.vstack((dy, dx, dh, dw)).transpose()

rois = torch.from_numpy(sample_roi).float()
roi_indices = 0 * np.ones((len(rois),), dtype=np.int32)
roi_indices = torch.from_numpy(roi_indices).float()

indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
indices_and_rois = xy_indices_and_rois.contiguous()

size = (7, 7)
adaptive_max_pool = nn.AdaptiveMaxPool2d((size[0], size[1]))
output = []
rois = indices_and_rois.data.float()
rois[:, 1:].mul_(1 / 16.0)
rois = rois.long()
num_rois = rois.size()[0]
for i in range(num_rois):
    roi = rois[i]
    im_idx = roi[0]
    im = out_map.narrow(0, im_idx, 1)[..., roi[2]:(roi[4] + 1), roi[1]:(roi[3] + 1)]
    output.append(adaptive_max_pool(im))
output = torch.cat(output, 0)
k = output.view(output.size()[0], -1)
roi_head_classifier = nn.Sequential(*[nn.Linear(25088, 4096), nn.Linear(4096, 4096)])
cls_loc = nn.Linear(4096, 21 * 4)
cls_loc.weight.data.normal_(0, 0.01)
cls_loc.bias.data.zero_()
score = nn.Linear(4096, 21)
k = roi_head_classifier(k)
roi_cls_loc = cls_loc(k)
roi_cls_score = score(k)

gt_roi_loc = torch.from_numpy(gt_roi_locs)
gt_roi_labels = torch.from_numpy(np.float32(gt_roi_labels)).long()
roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_labels, ignore_index=-1)
n_sample = roi_cls_loc.shape[0]
roi_loc = roi_cls_loc.view(n_sample, -1, 4)
roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_labels]
roi_loc_loss = calc_reg_loss(roi_loc, gt_roi_loc, gt_roi_labels)
roi_loss = roi_cls_loss + RCNN_LAMBDA * roi_loc_loss

total_loss = rpn_loss + roi_loss
