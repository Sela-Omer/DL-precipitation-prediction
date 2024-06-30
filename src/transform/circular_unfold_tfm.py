from typing import Callable

import torch


class CircularUnfoldTfm(Callable):
    def __init__(self, service, unfold_parameter, min_val, max_val):
        self.tfm_name = f'circular_unfold_{unfold_parameter}'
        self.service = service
        self.unfold_parameter_1 = f'{unfold_parameter}#sin'
        self.unfold_parameter_2 = f'{unfold_parameter}#cos'
        self.min_val = min_val
        self.max_val = max_val
        self.data_parameters = service.data_parameters.copy()

    def __call__(self, x):
        param_i1, param_i2 = self.service.get_parameter_index(
            self.unfold_parameter_1), self.service.get_parameter_index(self.unfold_parameter_2)

        param_normalized = (x[param_i1] - self.min_val) / (self.max_val - self.min_val)
        param_normalized = param_normalized * 2 * torch.pi

        x[param_i1] = torch.sin(param_normalized)
        x[param_i2] = torch.cos(param_normalized)

        return x
