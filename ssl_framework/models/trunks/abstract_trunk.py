import torch
import torch.nn as nn

from ssl_framework.models.trunks.utils import parse_out_keys_arg


class AbstractTrunk(nn.Module):
    def __init__(self):
        super(AbstractTrunk, self).__init__()

        self._feature_blocks = []
        self.all_feat_names = []

    def forward(self, x, out_feat_keys=None):
        out_feat_keys, max_out_feat = parse_out_keys_arg(
            out_feat_keys, self.all_feat_names
        )
        out_feats = [None] * len(out_feat_keys)

        feat = x
        prev_key = None
        for f in range(max_out_feat + 1):
            key = self.all_feat_names[f]

            if prev_key is not None and 'conv' in prev_key and 'fc' in key:
                feat = feat.view(feat.size(0), -1)
            feat = self._feature_blocks[f](feat)

            if key in out_feat_keys:
                out_feats[out_feat_keys.index(key)] = feat

            prev_key = key

        return out_feats