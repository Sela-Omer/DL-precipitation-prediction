from abc import ABC

import lightning as pl

from src.module.residual_nn import ResidualNN
from src.script.simple_nn_script import SimpleNNScript
from torchsummary import summary


class ResidualNNScript(SimpleNNScript, ABC):
    """
    A script for training a residual neural network model on meteorological data.

    """

    def create_architecture(self, datamodule: pl.LightningDataModule):
        example_input_array, _ = next(iter(datamodule.train_dataloader()))
        model_hyperparams = self.service.model_hyperparams if hasattr(self.service, 'model_hyperparams') else {}
        model = ResidualNN(self.service, example_input_array=example_input_array, **model_hyperparams)
        summary(model, input_size=example_input_array.shape[1:], device=str(model.device))
        return model
