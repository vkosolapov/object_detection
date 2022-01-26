import random
import numpy as np
import torch

from torch.utils.tensorboard import SummaryWriter
import wandb

from train_loop import TrainLoop
import albumentations as A
from albumentations.augmentations.transforms import CoarseDropout
from timm import create_model
from timm.models.resnet import _create_resnet, Bottleneck
from timm.models.sknet import SelectiveKernelBottleneck
from timm.models.resnest import ResNestBottleneck
from timm.models.res2net import Bottle2neck
from model import Model
import torch.nn as nn
import torch.optim as optim
from torchcontrib.optim import SWA
import torchmetrics

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

EXPERIMENT_NAME = "015_CenterNet_basic"
wandb.init(sync_tensorboard=True, project="object_detection_", name=EXPERIMENT_NAME)

if __name__ == "__main__":
    num_classes = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # backbone_model = _create_resnet("ecaresnet50d", False, **backbone_args)
    backbone_model = create_model("resnet50", num_classes=num_classes)
    model = Model(num_classes, backbone_model)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0001)
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
                    A.RandomResizedCrop(256, 256),
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
                p=1.0,
            ),
            A.OneOf(
                [
                    A.ElasticTransform(
                        alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03
                    ),
                    A.GridDistortion(),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ],
                p=0.5,
            ),
            A.OneOf(
                [A.GaussNoise(p=0.5), A.Blur(p=0.5), CoarseDropout(max_holes=5)], p=0.5
            ),
        ],
        p=1,
    )

    loop = TrainLoop(
        experiment_name=EXPERIMENT_NAME,
        device=device,
        workers=8,
        datadir="data/AFO/dataset.yaml",
        image_size=1280,
        batch_size=8,
        model=model,
        optimizer=optimizer,  # swa,
        num_epochs=500,
        criterion=nn.CrossEntropyLoss(num_classes=num_classes),
        metrics={},
        scheduler=scheduler,
        early_stopping=20,
        # checkpoint_file="checkpoints/checkpoint_139.pth",
    )

    loop.train_model()

