from typing import Callable

import torch


class NormalizeTfm(Callable):
    tfm_name = 'normalize'

    def __init__(self, service):
        self.service = service
        self.norm_parameters = service.norm_parameters

    def __call__(self, x):
        """
        Normalize the input data.
        :param x: The input data.
        :return: normalized data.
        """
        mean_tensor = torch.load(f'stats/{self.service.model_name}-mean.pt')
        std_tensor = torch.load(f'stats/{self.service.model_name}-std.pt')
        for param in self.norm_parameters:
            param_index_in_x = self.service.get_parameter_index(param)
            x[param_index_in_x] = (x[param_index_in_x] - mean_tensor[param_index_in_x]) / std_tensor[
                param_index_in_x]
        return x
