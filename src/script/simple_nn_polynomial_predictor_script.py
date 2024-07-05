from abc import ABC

import lightning as pl

from src.module.residual_nn import ResidualNN
from src.module.simple_nn_polynomial_predictor import SimpleNN_PolynomialPredictor
from src.module.simple_nn_skip_connection import SimpleNN_SkipConnection
from src.script.simple_nn_script import SimpleNNScript
from torchsummary import summary


class SimpleNNPolynomialPredictorScript(SimpleNNScript, ABC):
    """
    A script for training a simple neural network model with a polynomial predictor on meteorological data.

    """

    def create_architecture(self, datamodule: pl.LightningDataModule):
        example_input_array, _ = next(iter(datamodule.train_dataloader()))
        model_hyperparams = self.service.model_hyperparams if hasattr(self.service, 'model_hyperparams') else {}
        model = SimpleNN_PolynomialPredictor(self.service, example_input_array=example_input_array, **model_hyperparams)
        summary(model, input_size=example_input_array.shape[1:], device=str(model.device))
        return model
