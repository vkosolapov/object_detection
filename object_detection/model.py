import torch.nn as nn
from norm import CBatchNorm2d


class Model(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head
        prev_module = None
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                prev_module = module
            if isinstance(module, CBatchNorm2d):
                module.prev_module_weight = prev_module.weight

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
