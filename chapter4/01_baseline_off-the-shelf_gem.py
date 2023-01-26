import sys

import timm
import torch
import torch.nn as nn
from cirtorch.layers.pooling import GeM


class ResNetOfftheShelfGeM(nn.Module):
    def __init__(self, backbone="resnet34", pretrained=False):
        super(ResNetOfftheShelfGeM, self).__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
        )
        self.pooling = GeM()

    def forward(self, x):
        bs = x.size(0)
        x = self.backbone(x)[-1]
        x = self.pooling(x).view(bs, -1)
        return x


if __name__ == "__main__":
    model = ResNetOfftheShelfGeM(pretrained=True)
    x = torch.rand(4, 3, 244, 244)
    out = model(x)
    print(out.size())  # => torch.Size([4, 512])
