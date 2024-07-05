from typing import Any

import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from src.module.polynomial_predictor import PolynomialPredictor
from src.module.simple_nn import SimpleNN
from src.module.skip_connection_module import SkipConnectionModule
from src.service.service import Service


class SimpleNN_PolynomialPredictor(SimpleNN):
    """
    A simple neural network model with a polynomial predictor.

    """

    def __init__(self, service: Service, *args, **kwargs):
        degree = 5
        num_lookback_timesteps = service.lookback_range + 1
        super().__init__(service, *args, override_last_lin_planes=(degree + 1) * num_lookback_timesteps, **kwargs)

        polynomial_predictor = PolynomialPredictor(degree)

        self.layers = SkipConnectionModule(self.layers, polynomial_predictor, self._select_skip_connection,
                                           None)

    def _select_skip_connection(self, x):
        x = x.reshape(x.shape[0], len(self.input_parameters), self.lookback_range + 1)
        x = torch.index_select(x, 1, torch.tensor(self.target_parameter_indices).to(x.device))
        x = x.reshape(x.shape[0], -1)
        return x
