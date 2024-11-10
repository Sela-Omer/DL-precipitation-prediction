from typing import Any

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch import nn

from src.helper.param_helper import convert_param_to_type
from src.metric.mae_ci import MAEWithConfidenceInterval
from src.module.dropout_norm import DropoutNorm
from src.module.dynamic_cnn import DynamicCNN
from src.module.empty_norm import EmptyNorm
from src.module.half_batchnorm_2d import HalfBatchNorm2d
from src.service.service import Service


class CNNModule(pl.LightningModule):
    """
    :param service: A service object that contains the configuration parameters.

    """

    def __init__(self, service: Service, example_input_array=None, lr=1e-3, override_last_lin_planes=None, **kwargs):
        super().__init__()
        self.example_input_array = example_input_array
        self.service = service
        self.lr = lr

        self.mae_with_ci = MAEWithConfidenceInterval()

        self.lookback_range = service.lookback_range
        self.forecast_range = service.forecast_range

        self.data_parameters = service.data_parameters
        self.target_parameters = service.target_parameters
        self.input_parameters = service.input_parameters
        self.input_parameter_indices = [service.get_parameter_index(input_param) for input_param in
                                        self.input_parameters]
        self.target_parameter_indices = [service.get_parameter_index(target_param) for target_param in
                                         self.target_parameters]

        in_channels = len(self.input_parameters) * (self.lookback_range + 1)
        norm_layer_dict = {'BATCH_NORM': nn.BatchNorm2d, 'HALF_BATCH_NORM': HalfBatchNorm2d,
                           'EMPTY_NORM': EmptyNorm, 'DROPOUT_NORM': DropoutNorm}
        norm_layer = norm_layer_dict[service.config['APP']['NORM_LAYER']]
        pool_layer_dict = {'AVG_POOL': nn.AvgPool2d, 'MAX_POOL': nn.MaxPool2d, }
        pool_layer = pool_layer_dict[service.config['APP']['POOL_LAYER']]

        lin_ch_mult = convert_param_to_type(self.service.config['APP']['LIN_CH_MULT']) if 'LIN_CH_MULT' in \
                                                                                          self.service.config[
                                                                                              'APP'] else 2

        num_classes = len(self.target_parameters) if override_last_lin_planes is None else override_last_lin_planes
        self.cnn = DynamicCNN(norm_layer, pool_layer, in_channels, num_classes, service.network_depth,
                              lin_ch_mult=lin_ch_mult)

    def forward(self, x) -> Any:
        """
        Forward pass of the model.
        :param x: The input data.
        :return: The output of the model.
        """
        assert len(x.shape) == 5, f"The input data must be 5D. Instead got shape: {x.shape}"
        x = torch.index_select(x, 1, torch.tensor(self.input_parameter_indices).to(x.device))
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        return self.cnn(x)

    def _generic_step(self, batch, step_name: str) -> STEP_OUTPUT:
        """
        A generic step for the model.
        :param batch: The batch of data.
        :param step_name: The name of the step.
        :return: The loss for the batch.
        """
        X, y = batch
        # i = np.random.randint(0, 1000)

        # # check is X has nan or inf or -inf
        # if torch.isnan(X).any() or torch.isinf(X).any():
        #     print(f'X {i} has nan or inf or -inf values.')

        X = torch.index_select(X, 1, torch.tensor(self.input_parameter_indices).to(X.device))
        y = torch.index_select(y, 1, torch.tensor(self.target_parameter_indices).to(y.device))

        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2], X.shape[3], X.shape[4])
        y = y.reshape(y.shape[0], y.shape[1] * y.shape[2], y.shape[3], y.shape[4])

        y = y[..., y.shape[-2] // 2, y.shape[-1] // 2]

        y_hat = self.cnn(X)

        loss = nn.functional.mse_loss(y_hat, y)
        mae = nn.functional.l1_loss(y_hat, y)

        # # check if the loss is nan or inf or -inf or num of elements is 0
        # if torch.isnan(loss) or torch.isinf(loss) or loss.numel() == 0:
        #     print(f'Loss {i} is {loss}.')
        #     # self.log(f'{step_name}_loss', 0, sync_dist=True)
        #     # self.log(f'{step_name}_mae', 0, prog_bar=True, sync_dist=True)
        #     return None

        self.mae_with_ci.update(y_hat, y)
        self.log(f'{step_name}_loss', loss, sync_dist=True)
        self.log(f'{step_name}_mae', mae, prog_bar=True, sync_dist=True)

        if len(self.target_parameters) > 1:
            for i, param in enumerate(self.target_parameters):
                yi = y[:, i]
                yi_hat = y_hat[:, i]
                mae_i = nn.functional.l1_loss(yi_hat, yi)
                self.log(f'{step_name}_mae_{param}', mae_i, prog_bar=True, sync_dist=True)

        return loss

    def _generic_epoch_end(self, epoch_name: str):
        mae_ci_dict = self.mae_with_ci.compute()
        for name, value in mae_ci_dict.items():
            self.log(f'{epoch_name}_{name}', value, sync_dist=True)
        self.mae_with_ci.reset()

    # def training_epoch_end(self, outputs) -> None:
    #     self._generic_epoch_end(outputs, 'train')

    # def validation_epoch_end(self, outputs) -> None:
    #     self._generic_epoch_end(outputs, 'val')

    def on_test_epoch_end(self) -> None:
        self._generic_epoch_end('test')

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
