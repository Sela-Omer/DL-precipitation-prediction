from types import NoneType
from typing import Union

import torch
from torch import nn


class SkipConnectionModule(nn.Module):
    """
    A module that implements a skip connection between two modules.
    :arg module_1: The first module.
    :arg module_2: The second module.
    :arg select_skip_connection_function: A function that selects the skip connection.
    :arg concat_dim: The dimension to concatenate the skip connection on.

    """

    def __init__(self, module_1: nn.Module, module_2: nn.Module, select_skip_connection_function,
                 concat_dim: Union[int, NoneType]):
        super().__init__()
        self.module_1 = module_1
        self.module_2 = module_2
        self.select_skip_connection_function = select_skip_connection_function
        self.concat_dim = concat_dim

    def forward(self, x):
        skip = self.select_skip_connection_function(x)
        x = self.module_1(x)
        if self.concat_dim is None:
            x = self.module_2(x, skip)
        else:
            x = torch.cat([x, skip], dim=self.concat_dim)
            x = self.module_2(x)
        return x
