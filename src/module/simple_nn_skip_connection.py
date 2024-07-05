from typing import Any

import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from src.module.simple_nn import SimpleNN
from src.module.skip_connection_module import SkipConnectionModule
from src.service.service import Service


class SimpleNN_SkipConnection(SimpleNN):
    """
    A simple neural network model with a skip connection.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        body = list(self.layers[:-1])
        body_out_features = self.layers[-1].in_features
        skip_len = (self.lookback_range + 1) * len(self.target_parameters)
        last_lin = nn.Linear(body_out_features + skip_len,
                             len(self.target_parameters))
        nn.init.constant_(last_lin.weight, 0)

        # Set the last k weights to 1/k
        with torch.no_grad():
            last_lin.weight[:, -skip_len:] = 1 / skip_len
        # Initialize all biases to 0
        nn.init.constant_(last_lin.bias, 0)

        self.layers = SkipConnectionModule(nn.Sequential(*body), nn.Sequential(last_lin), self._select_skip_connection,
                                           1)

    def _select_skip_connection(self, x):
        x = x.reshape(x.shape[0], len(self.input_parameters), self.lookback_range + 1)
        x = torch.index_select(x, 1, torch.tensor(self.target_parameter_indices).to(x.device))
        for t in range(x.shape[2] - 1):
            prv = x[:, :, t]
            nxt = x[:, :, -1]
            times = x.shape[2] - t - 1
            x[:, :, t] = (nxt + ((self.forecast_range) * (nxt - prv)) / times)
        return x.reshape(x.shape[0], -1)
