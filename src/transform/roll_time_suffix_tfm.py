import torch

from src.transform.op_suffix_tfm import OpSuffixTfm


class RollTimeSuffixTfm(OpSuffixTfm):
    def __init__(self, service, roll_time: int, suffix):
        super().__init__(service, f'roll_time_{roll_time}', suffix)
        self.roll_time = roll_time

    def feature_operation(self, x_feature, x_feature_matching_no_suffix=None):
        # TIMES x HEIGHT x WIDTH
        return torch.roll(x_feature, shifts=self.roll_time, dims=0)
