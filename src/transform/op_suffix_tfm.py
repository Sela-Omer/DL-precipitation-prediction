from abc import abstractmethod
from typing import Callable

import torch


class OpSuffixTfm(Callable):
    def __init__(self, service, opname, suffix):
        self.tfm_name = f'op={opname}_suffix={suffix}'
        self.service = service
        self.data_parameters = service.data_parameters.copy()
        self.suffix = suffix

    @abstractmethod
    def feature_operation(self, x_feature, x_feature_matching_no_suffix=None):
        # TIMES x HEIGHT x WIDTH
        pass

    def __call__(self, x):
        # PARAMS x TIMES x HEIGHT x WIDTH
        assert len(x.shape) == 4, f"shape of x must be (PARAMS x TIMES x HEIGHT x WIDTH) instead got {x.shape}"

        for param in self.data_parameters:
            if self.suffix in param:
                param_i = self.service.get_parameter_index(param)

                # find matching parameter without suffix
                param_matching_no_suffix = param.replace(self.suffix, '')
                param_i_matching_no_suffix = self.service.get_parameter_index(
                    param_matching_no_suffix) if param_matching_no_suffix in self.data_parameters else None

                x_param = x[param_i]
                x_param_no_suffix = x[param_i_matching_no_suffix] if param_i_matching_no_suffix is not None else None

                x[param_i] = self.feature_operation(x_param, x_feature_matching_no_suffix=x_param_no_suffix)

        return x
