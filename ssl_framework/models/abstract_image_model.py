import torch.nn as nn
from abc import ABC, abstractmethod

from ssl_framework.config.default import cfg
from ssl_framework.models.heads import MLP
from ssl_framework.models.trunks import TRUNKS


class AbstractImageModel(ABC):
    def __init__(self):
        super(AbstractImageModel, self).__init__()

        self.eval_mode = cfg.MODEL.FEATURE_EVAL_MODE
        self.trunk = TRUNKS[cfg.MODEL.TRUNK.NAME]()
        self.heads = nn.ModuleList()

        for kwargs in cfg.MODEL.HEAD.PARAMS:
            self.heads.append(MLP(**kwargs))

    def vanilla_forward(self, x, out_feat_keys=None):
        out_feats = self.trunk(x, out_feat_keys)

        if self.heads is None:
            return out_feats

        assert len(self.heads) == len(out_feats),\
               "number of heads should be equal to number of out_feat_keys"

        clf_outs = []
        for i in range(len(out_feats)):
            clf_outs.append(self.heads[i](out_feats[i]))

        return clf_outs

    @abstractmethod
    def forward(self, x, out_feat_keys=None):
        pass

    @abstractmethod
    def loss(self, logits, **kwargs):
        """
        compute loss according to model specifications
        """
        pass