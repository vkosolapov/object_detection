import torch.nn as nn
from timm.models.layers import SelectAdaptivePool2d


class TIMMBackbone(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        children = list(backbone.children())
        children.reverse()
        for idx, module in enumerate(children):
            if isinstance(module, SelectAdaptivePool2d):
                backbone = nn.Sequential(*list(backbone.children())[: -(idx + 1)])
                break
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)
