import torch.nn as nn

from backbone.timm_backbone import TIMMBackbone
from head.centernet import CenterNet


class Model(nn.Module):
    def __init__(self, num_classes, backbone):
        super().__init__()
        self.backbone = TIMMBackbone(backbone)
        self.head = CenterNet(num_classes, backbone)
                
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
