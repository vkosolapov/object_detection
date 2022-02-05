import torch.nn as nn


class CenterNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        channels = 0
        modules = list(backbone.modules())
        modules.reverse()
        for module in modules:
            if isinstance(module, nn.Conv2d):
                channels = module.out_channels
                break
        self.channels = channels

        self.decoder = self._make_decoder(
            num_layers=3, channels_list=[256, 128, 64], kernels_list=[4, 4, 4],
        )

        self.classification_head = self._make_head(
            input_channels=64, output_channels=num_classes
        )
        self.size_head = self._make_head(input_channels=64, output_channels=2)
        self.offset_head = self._make_head(input_channels=64, output_channels=2)

    def _make_decoder(self, num_layers, channels_list, kernels_list):
        layers = []
        for i in range(num_layers):
            channels = channels_list[i]
            kernel = kernels_list[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.channels,
                    out_channels=channels,
                    kernel_size=kernel,
                    stride=2,
                    padding=1,
                    output_padding=0,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
            self.channels = channels
        return nn.Sequential(*layers)

    def _make_head(self, input_channels, output_channels):
        return nn.Sequential(
            nn.Conv2d(
                input_channels, input_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                input_channels, output_channels, kernel_size=1, stride=1, padding=0
            ),
        )

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.decoder(x)
        cls = self.classification_head(x)
        size = self.size_head(x)
        offset = self.offset_head(x)
        return (cls, size, offset)
