from typing import Any

import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from src.module.polynomial_predictor import PolynomialPredictor
from src.module.rnn_predictor import RNNPredictor
from src.module.simple_nn import SimpleNN
from src.module.skip_connection_module import SkipConnectionModule
from src.service.service import Service


class SimpleNN_RNNPredictor(SimpleNN):
    """
    A simple neural network model with RNN predictor.

    """

    def __init__(self, service: Service, *args, **kwargs):
        hidden_size = 128
        super().__init__(service, *args, override_last_lin_planes=hidden_size * 5 + 1 + hidden_size * hidden_size,
                         **kwargs)

        rnn_predictor = RNNPredictor(hidden_size)

        self.layers = SkipConnectionModule(self.layers, rnn_predictor, self._select_skip_connection,
                                           None)

    def _select_skip_connection(self, x):
        x = x.reshape(x.shape[0], len(self.input_parameters), self.lookback_range + 1)
        x = torch.index_select(x, 1, torch.tensor(self.target_parameter_indices).to(x.device))
        x = x.reshape(x.shape[0], -1)
        return x
