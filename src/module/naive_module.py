from typing import Any

import lightning as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from src.service.service import Service


class NaiveModule(pl.LightningModule):
    """
    A naive_module model.
    for lookback=0 returns the last value in the time series as the prediction.
    for lookback>0 returns the lerps first and last time series as the prediction.
    :param service: A service object that contains the configuration parameters.

    """

    def __init__(self, service: Service, example_input_array=None, **_):
        super().__init__()
        self.service = service
        self.example_input_array = example_input_array
        self.lookback_range = service.lookback_range
        self.forecast_range = service.forecast_range

        assert len(service.target_parameters) == 1, "The NaiveModule only supports one target parameter."
        self.target_parameter = service.target_parameters[0]
        self.target_parameter_index = service.get_parameter_index(self.target_parameter)
        self.dummy_param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x) -> Any:
        x = x[:, self.target_parameter_index]
        if x.shape[1] == 1:
            return x
        else:
            prv = x[:, 0]
            nxt = x[:, -1]
            times = x.shape[1] - 1
            return (nxt + ((self.forecast_range) * (nxt - prv)) / times)[..., None]

    def _generic_step(self, batch, step_name: str) -> STEP_OUTPUT:
        X, y = batch

        y_hat = self.forward(X)
        y = y[:, self.target_parameter_index]

        assert len(
            y_hat.shape) == 2, f"The input data must be 2D after squeeze of time dim. instead got shape: {y_hat.shape}"
        assert len(y.shape) == 2, f"The target data must be 2D after squeeze of time dim. instead got shape: {y.shape}"

        loss = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)
        self.log(f'{step_name}_loss', loss, sync_dist=True)
        self.log(f'{step_name}_mae', mae, prog_bar=True, sync_dist=True)

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
        return None
        # optimizer = torch.optim.AdamW(self.parameters(),
        #                               lr=1e-3)
        # return optimizer
