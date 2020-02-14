"""
taken from https://github.com/facebookresearch/fair-sslime/blob/master/sslime/utils/utils.py
"""


def parse_out_keys_arg(out_feat_keys, all_feat_names):
    """
    Checks if all out_feature_keys are mapped to a layer in the model.
    Ensures no duplicate features are requested.
    Returns the last layer to forward pass through for efficiency.
    Adapted from (https://github.com/gidariss/FeatureLearningRotNet)
    """

    # if len(out_feat_keys) == 0, then None
    if out_feat_keys is not None and len(out_feat_keys) == 0:
        out_feat_keys = None

    # By default return the features of the last layer / module.
    out_feat_keys = [all_feat_names[-1]] if out_feat_keys is None else out_feat_keys

    for f, key in enumerate(out_feat_keys):
        if key not in all_feat_names:
            raise ValueError(
                "Feature with name {0} does not exist. Existing features: {1}.".format(
                    key, all_feat_names
                )
            )
        elif key in out_feat_keys[:f]:
            raise ValueError("Duplicate output feature key: {0}.".format(key))

    # Find the highest output feature in `out_feat_keys
    max_out_feat = max(all_feat_names.index(key) for key in out_feat_keys)

    return out_feat_keys, max_out_feat