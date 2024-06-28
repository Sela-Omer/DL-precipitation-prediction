from typing import Any

import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from src.module.residual_nn_block import ResidualNNBlock
from src.module.simple_nn import SimpleNN
from src.service.service import Service


class ResidualNN(SimpleNN):
    """
    A residual neural network model.

    """

    def _make_layer(self, in_features, out_features):
        """
        Create a layer with a linear transformation followed by a ReLU activation function.
        :param in_features:
        :param out_features:
        :return:
        """
        return ResidualNNBlock(in_features, out_features)
