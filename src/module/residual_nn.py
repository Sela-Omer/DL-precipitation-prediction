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

    def _make_layer(self, in_features, out_features, i):
        """
        Create a layer with a linear transformation followed by a ReLU activation function.
        :param in_features:
        :param out_features:
        :return:
        """
        if self.service.config['FIT']['DROPOUT'] == 'True' and self.network_depth // 4 <= i < (
                self.network_depth * 3) // 4:
            return ResidualNNBlock(in_features, out_features, dropout_pct=0.25)
        return ResidualNNBlock(in_features, out_features)
