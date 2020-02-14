"""
adapted from: https://github.com/bbrattoli/JigsawPuzzlePytorch/blob/master/JigsawNetwork.py
"""
import torch
import torch.nn as nn

from ssl_framework.config.default import cfg
from ssl_framework.models.trunks.abstract_trunk import AbstractTrunk


class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class JigsawAlexNetTrunk(AbstractTrunk):
    def __init__(self):
        super(JigsawAlexNetTrunk, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(local_size=5, alpha=0.0001, beta=0.75)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            LRN(local_size=5, alpha=0.0001, beta=0.75)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.fc6 = nn.Sequential(
            nn.Linear(256 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

        self._feature_blocks = nn.ModuleList(
            [self.conv1, self.conv2, self.conv3,
             self.conv4, self.conv5, self.fc6]
        )

        self.all_feat_names = [
            "conv1",
            "conv2",
            "conv3",
            "conv4",
            "conv5",
            "fc6",
        ]
        assert len(self.all_feat_names) == len(self._feature_blocks)