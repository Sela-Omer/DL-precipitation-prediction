from typing import Any

import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from src.service.service import Service


class SimpleNN(pl.LightningModule):
    """
    A simple neural network model.
    :param service: A service object that contains the configuration parameters.

    """

    def __init__(self, service: Service, example_input_array=None, lr=1e-3,
                 override_last_lin_planes=None,
                 **kwargs):
        super().__init__()
        self.example_input_array = example_input_array
        self.service = service
        self.lr = lr

        self.lookback_range = service.lookback_range
        self.forecast_range = service.forecast_range

        # assert self.lookback_range == 0, "The lookback range must be 0 for the SimpleNN model. As there is no temporal component in the architecture."

        self.data_parameters = service.data_parameters
        self.target_parameters = service.target_parameters
        self.input_parameters = service.input_parameters
        self.input_parameter_indices = [service.get_parameter_index(input_param) for input_param in
                                        self.input_parameters]
        self.target_parameter_indices = [service.get_parameter_index(target_param) for target_param in
                                         self.target_parameters]

        self.network_depth = service.network_depth

        layer_lst = []
        layer_planes = [len(self.input_parameters) * (self.lookback_range + 1) * (2 ** i) for i in
                        range(self.network_depth)]
        for i in range(self.network_depth - 1):
            layer_lst.append(self._make_layer(layer_planes[i], layer_planes[i + 1], i))

        last_lin_planes = override_last_lin_planes if override_last_lin_planes is not None else len(
            self.target_parameters)
        last_lin = nn.Linear(layer_planes[-1], last_lin_planes)
        layer_lst.append(last_lin)

        self.layers = nn.Sequential(*layer_lst)

    def _make_layer(self, in_features, out_features, i):
        """
        Create a layer with a linear transformation followed by a ReLU activation function.
        :param in_features:
        :param out_features:
        :return:
        """
        layer = [nn.Linear(in_features, out_features), nn.ReLU()]
        if self.service.config['FIT']['DROPOUT'] == 'True' and self.network_depth // 4 <= i < (
                self.network_depth * 3) // 4:
            layer.append(nn.Dropout(0.1))
        return nn.Sequential(*layer)

    def forward(self, x) -> Any:
        """
        Forward pass of the model.
        :param x: The input data.
        :return: The output of the model.
        """
        x = torch.index_select(x, 1, torch.tensor(self.input_parameter_indices).to(x.device))
        x = x.reshape(x.shape[0], -1)
        return self.layers(x)

    def _generic_step(self, batch, step_name: str) -> STEP_OUTPUT:
        """
        A generic step for the model.
        :param batch: The batch of data.
        :param step_name: The name of the step.
        :return: The loss for the batch.
        """
        X, y = batch

        X = torch.index_select(X, 1, torch.tensor(self.input_parameter_indices).to(X.device))
        y = torch.index_select(y, 1, torch.tensor(self.target_parameter_indices).to(y.device))

        X = X.reshape(X.shape[0], -1)
        y = y.squeeze(-1)
        assert len(y.shape) == 2, f"The target data must be 2D after squeeze of time dim. instead got shape: {y.shape}"

        y_hat = self.layers(X)

        loss = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        self.log(f'{step_name}_loss', loss, sync_dist=True)
        self.log(f'{step_name}_mae', mae, prog_bar=True, sync_dist=True)
        return loss

    def training_step(self, batch) -> STEP_OUTPUT:
        """
        The training step for the model.
        :param batch: The batch of data.
        :return: The loss for the batch.
        """
        return self._generic_step(batch, 'train')

    def validation_step(self, batch) -> STEP_OUTPUT:
        """
        The validation step for the model.
        :param batch: The batch of data.
        :return: The loss for the batch.
        """
        return self._generic_step(batch, 'val')

    def test_step(self, batch) -> STEP_OUTPUT:
        """
        The test step for the model.
        :param batch: The batch of data.
        :return: The loss for the batch.
        """
        return self._generic_step(batch, 'test')

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure the optimizer and learning rate scheduler.
        :return: The optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr)
        return optimizer
