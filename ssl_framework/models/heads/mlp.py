import math
import torch
import torch.nn as nn

from ssl_framework.config.default import cfg


class MLP(nn.Module):
    def __init__(self, dims, dropout, dropout_probs, in_channels=0):
        """
        length of dropout and dropout_probs should be 2 less than length
        of dims
        """
        super(MLP, self).__init__()

        self.eval_mode = cfg.MODEL.FEATURE_EVAL_MODE
        self.in_channels = in_channels
        self.pool_type = cfg.MODEL.HEAD.POOL

        self.batchnorm = cfg.MODEL.HEAD.APPLY_BATCHNORM
        self.bn_eps = cfg.MODEL.HEAD.BATCHNORM_EPS
        self.bn_momentum = cfg.MODEL.HEAD.BATCHNORM_MOMENTUM

        self.dims = dims
        self.dropout = dropout
        self.dropout_probs = dropout_probs

        assert len(dims) - 2 == len(dropout) and len(dims) - 2 == len(dropout_probs),\
               "length of dropout/dropout_probs should be 2 less than length of dims"

        if self.eval_mode:
            assert self.in_channels != 0,\
                   "in_channels should not be 0 in eval mode"

            resize_to = int(math.sqrt(self.dims[0] // self.in_channels))

            if self.pool_type == 'avg':
                self.pool = nn.AdaptiveAvgPool2d((resize_to, resize_to))
            elif self.pool_type == 'max':
                self.pool = nn.AdaptiveMaxPool2d((resize_to, resize_to))

            if self.batchnorm:
                self.spatial_bn = nn.BatchNorm2d(self.in_channels,
                                                 eps=self.bn_eps,
                                                 momentum=self.bn_momentum)

        self.clf = self.create_mlp()

    def create_mlp(self):
        layers = []
        last_dim = self.dims[0]

        dropout_idx = 0
        for dim in self.dims[1:-1]:
            layers.append(nn.Linear(last_dim, dim))

            if self.batchnorm:
                layers.append(
                    nn.BatchNorm1d(
                        dim,
                        eps=self.bn_eps,
                        momentum=self.bn_momentum,
                    )
                )

            layers.append(nn.ReLU(inplace=True))

            if self.dropout[dropout_idx]:
                layers.append(nn.Dropout(p=self.dropout_probs[dropout_idx]))
            dropout_idx += 1

            last_dim = dim

        layers.append(nn.Linear(last_dim, self.dims[-1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.eval_mode:
            x = self.pool(x)

            if self.batchnorm:
                x = self.spatial_bn(x)

        if len(x.size()) != 2:
            x = x.view(x.size(0), -1)

        x = self.clf(x)
        return x
