import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_framework.config.default import cfg
from ssl_framework.models.abstract_image_model import AbstractImageModel


class JigsawImageModel(AbstractImageModel, nn.Module):
    def __init__(self):
        super(JigsawImageModel, self).__init__()

    # TODO: optimize training mode for single output from trunk
    # ie, out_feat_keys = []
    def forward(self, x, out_feat_keys=None):
        """
        x is a list of image patches
        x: [B, T, C, H, W]
        """
        if self.eval_mode:
            return self.vanilla_forward(x, out_feat_keys)

        # output the last layer from trunk if not in eval mode
        out_feat_keys = []

        # create a multiple level out_feats list
        B, T, C, H, W = x.size()
        x = x.transpose(0, 1).contiguous()

        # out_feats is a T x len(out_feat_keys) size matrix
        out_feats = []
        num_out_feats = None
        for i in range(T):
            # layer outputs for a single patch
            patch_feats = self.trunk(x[i], out_feat_keys)

            if num_out_feats is None:
                num_out_feats = len(patch_feats)

            out_feats.append(patch_feats)

        assert len(self.heads) == len(out_feats[0]),\
               "number of heads should be equal to number of out_feat_keys"

        clf_outs = []
        for i in range(num_out_feats):
            patch_feats = []

            for j in range(T):
                patch_feats.append(out_feats[j][i])
            patch_feats = torch.stack(patch_feats, dim=0)
            eval_feats = patch_feats.transpose(0, 1).contiguous()

            eval_feats = self.heads[i](eval_feats)
            clf_outs.append(eval_feats)

        return clf_outs

    def loss(self, logits, **kwargs):
        """
        input: logits output from the network
        target: target values from ground-truth
        """
        assert 'target' in kwargs.keys(), "pass target argument for this model"

        output = F.log_softmax(logits, dim=1)
        return F.nll_loss(output, kwargs['target']), output