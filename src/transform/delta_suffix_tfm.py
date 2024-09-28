import torch

from src.transform.op_suffix_tfm import OpSuffixTfm


class DeltaSuffixTfm(OpSuffixTfm):
    def __init__(self, service, suffix):
        super().__init__(service, f'delta', suffix)

    def feature_operation(self, x_feature, x_feature_matching_no_suffix=None):
        # TIMES x ...
        return x_feature - x_feature_matching_no_suffix
