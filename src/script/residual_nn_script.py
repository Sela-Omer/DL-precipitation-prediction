from abc import ABC

import lightning as pl

from src.module.residual_nn import ResidualNN
from src.script.simple_nn_script import SimpleNNScript


class ResidualNNScript(SimpleNNScript, ABC):
    """
    A script for training a residual neural network model on meteorological data.

    """

    def create_architecture(self, datamodule: pl.LightningDataModule):
        example_input_array, _ = next(iter(datamodule.train_dataloader()))
        model_hyperparams = self.service.model_hyperparams if hasattr(self.service, 'model_hyperparams') else {}
        return ResidualNN(self.service, example_input_array=example_input_array, **model_hyperparams)
