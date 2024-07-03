from typing import Any

import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn
from torchvision.models import resnet34

from src.service.service import Service


class ResNetModule(pl.LightningModule):
    """
    :param service: A service object that contains the configuration parameters.

    """

    def __init__(self, service: Service, example_input_array=None, lr=1e-3, **kwargs):
        super().__init__()
        self.example_input_array = example_input_array
        self.service = service
        self.lr = lr

        self.lookback_range = service.lookback_range

        self.data_parameters = service.data_parameters
        self.target_parameters = service.target_parameters
        self.input_parameters = service.input_parameters
        self.input_parameter_indices = [service.get_parameter_index(input_param) for input_param in
                                        self.input_parameters]
        self.target_parameter_indices = [service.get_parameter_index(target_param) for target_param in
                                         self.target_parameters]

        in_channels = len(self.input_parameters) * (self.lookback_range + 1)
        self.resnet34_model = resnet34(pretrained=False, num_classes=len(self.target_parameters))
        conv_1_orig = self.resnet34_model.conv1
        self.resnet34_model.conv1 = nn.Conv2d(in_channels, conv_1_orig.out_channels, kernel_size=conv_1_orig.kernel_size,
                                         stride=conv_1_orig.stride, padding=conv_1_orig.padding, bias=conv_1_orig.bias)

    def forward(self, x) -> Any:
        """
        Forward pass of the model.
        :param x: The input data.
        :return: The output of the model.
        """
        assert len(x.shape) == 5, f"The input data must be 5D. Instead got shape: {x.shape}"
        x = torch.index_select(x, 1, torch.tensor(self.input_parameter_indices).to(x.device))
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2], x.shape[3], x.shape[4])
        return self.resnet34_model(x)

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

        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2], X.shape[3], X.shape[4])
        y = y.reshape(y.shape[0], y.shape[1]*y.shape[2], y.shape[3], y.shape[4])

        y = y[..., y.shape[-2]//2, y.shape[-1]//2]

        y_hat = self.resnet34_model(X)

        loss = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        self.log(f'{step_name}_loss', loss)
        self.log(f'{step_name}_mae', mae, prog_bar=True)
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
