"""
taken from https://github.com/facebookresearch/fair-sslime/blob/master/sslime/models/trunks/resnet50.py
"""
import torch.nn as nn
import torchvision.models as models

from ssl_framework.config.default import cfg
from ssl_framework.models.trunks.abstract_trunk import AbstractTrunk


class Resnet_50(AbstractTrunk):
    def __init__(self):
        super(Resnet_50, self).__init__()
        model = models.resnet50()
        conv1 = nn.Sequential(model.conv1, model.bn1, model.relu)

        self._feature_blocks = nn.ModuleList(
            [
                conv1,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                model.avgpool
            ]
        )

        self.all_feat_names = [
            "conv1",
            "res1",
            "res2",
            "res3",
            "res4",
            "res5",
            "res5avg"
        ]

        assert len(self.all_feat_names) == len(self._feature_blocks)