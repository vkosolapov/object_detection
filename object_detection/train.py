from functools import partial
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from torch.utils.tensorboard import SummaryWriter
import wandb

from train_loop import TrainLoop
import albumentations as A
from albumentations.augmentations.transforms import CoarseDropout
from dataloader import augment
from timm import create_model
from timm.models.resnet import _create_resnet
from timm.models.resnest import ResNestBottleneck
from backbone import TIMMBackbone
from model import Model
from loss import LabelSmoothingFocalLoss, RegressionLossWithMask
import torch.optim as optim
from torchcontrib.optim import SWA
from torchmetrics.detection.map import MeanAveragePrecision

from detectors.centernet.dataset import CenternetDataset, postprocess_predictions
from detectors.centernet.model import CenterNet
from detectors.centernet.loss import compute_losses

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

EXPERIMENT_NAME = "018_Augmentations_basic"
wandb.init(sync_tensorboard=True, project="object_detection_", name=EXPERIMENT_NAME)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = 4
    datadir = "data/AFO/PART_1/PART_1"
    num_classes = 6
    image_size = 640
    batch_size = 32
    num_epochs = 500
    early_stopping = 100
    learning_rate = 0.01
    checkpoint_file = None  # "checkpoints/checkpoint_139.pth",

    augmentations = A.Compose(
        [
            A.OneOf(
                [
                    # A.Affine(
                    #    scale=(-0.1, 0.1),
                    #    translate_percent=(-0.0625, 0.0625),
                    #    rotate=(-45, 45),
                    #    shear=(-15, 15),
                    # ),
                    A.ShiftScaleRotate(),
                    A.RandomResizedCrop(image_size, image_size),
                    A.HorizontalFlip(),
                    # A.VerticalFlip(),
                    # A.Transpose(),
                ],
                p=1.0,
            ),
            A.OneOf(
                [
                    A.RandomGamma(),
                    A.RGBShift(),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1),
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20,
                    ),
                ],
                p=0.0,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                    ),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ],
                p=0.0,
            ),
            A.OneOf(
                [A.GaussNoise(p=0.5), A.Blur(p=0.5), CoarseDropout(max_holes=5)], p=0.0
            ),
        ],
        p=1,
        bbox_params=A.BboxParams(format="pascal_voc", min_area=16, min_visibility=0.1),
    )

    datasets = {
        "train": CenternetDataset(
            datadir, "train", num_classes, image_size, augmentations
        ),
        "val": CenternetDataset(datadir, "val", num_classes, image_size),
    }

    backbone_args = dict(
        block=ResNestBottleneck,
        layers=[2, 2, 2, 2],  # [3, 4, 6, 3],
        cardinality=32,
        base_width=4,
        # block_args=dict(attn_layer="se", sk_kwargs=dict(split_input=True), scale=4), # SKNet
        block_args=dict(radix=2, avd=True, avd_first=False),  # ResNeSt
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        num_classes=num_classes,
    )
    backbone_model = _create_resnet("ecaresnet50d", False, **backbone_args)
    backbone_model = create_model("resnet18", num_classes=num_classes, pretrained=True)
    backbone_model = TIMMBackbone(backbone_model)
    head_model = CenterNet(backbone_model, num_classes)
    model = Model(backbone_model, head_model)
    model = model.to(device)

    criterion = {
        "cls": LabelSmoothingFocalLoss(
            num_classes=num_classes,
            one_hot_label_format=True,
            gamma=2.0,
            smoothing=0.1,
        ),
        "size": RegressionLossWithMask(smooth=True),
        "offset": RegressionLossWithMask(smooth=True),
    }
    criterion_weights = {
        "cls": 10.0,
        "size": 0.1,
        "offset": 1.0,
    }

    losses_computer = compute_losses
    predictions_postprocessor = postprocess_predictions

    metrics = {
        "mAP@0.5": MeanAveragePrecision(
            box_format="xyxy",
            iou_thresholds=[0.5],
            rec_thresholds=[0.0],
            max_detection_thresholds=[100],
            class_metrics=False,
        ),
        "mAP@0.5:0.95": MeanAveragePrecision(
            box_format="xyxy",
            iou_thresholds=[(50.0 + th * 5.0) / 100.0 for th in range(10)],
            rec_thresholds=[0.0],
            max_detection_thresholds=[100],
            class_metrics=False,
        ),
    }
    main_metric = "mAP@0.5:0.95"

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    # swa = SWA(optimizer_conv, swa_start=10, swa_freq=5, swa_lr=0.05)
    swa = SWA(optimizer)
    # scheduler = CyclicCosineDecayLR(
    #    optimizer,
    #    warmup_epochs=5,
    #    warmup_start_lr=0.005,
    #    warmup_linear=False,
    #    init_decay_epochs=5,
    #    min_decay_lr=0.001,
    #    restart_lr=0.01,
    #    restart_interval=10,
    #    # restart_interval_multiplier=1.2,
    # )
    scheduler = None

    grad_init = {
        "gradinit_lr": 1e-3,
        "gradinit_iters": 300,
        "gradinit_alg": "adam",
        "gradinit_eta": 0.1,
        "gradinit_min_scale": 0.01,
        "gradinit_grad_clip": 1,
        "gradinit_gamma": float("inf"),
        "gradinit_normalize_grad": False,
        "gradinit_resume": "",
        "gradinit_bsize": -1,
        "batch_no_overlap": False,
        "expname": "default",
    }

    loop = TrainLoop(
        experiment_name=EXPERIMENT_NAME,
        device=device,
        workers=workers,
        datasets=datasets,
        image_size=image_size,
        batch_size=batch_size,
        model=model,
        optimizer=optimizer,  # swa,
        num_epochs=num_epochs,
        criterion=criterion,
        criterion_weights=criterion_weights,
        losses_computer=losses_computer,
        predictions_postprocessor=predictions_postprocessor,
        metrics=metrics,
        main_metric=main_metric,
        scheduler=scheduler,
        early_stopping=early_stopping,
        checkpoint_file=checkpoint_file,
    )

    loop.train_model()

