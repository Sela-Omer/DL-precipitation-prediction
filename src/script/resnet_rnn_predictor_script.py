from abc import ABC

import lightning as pl
from torchsummary import summary

from src.datamodule.meteorological_data_module import MeteorologicalDataModule
from src.dataset.cnn_meteorological_dataset import CNN_MeteorologicalDataset
from src.dataset.nn_meteorological_dataset import NN_MeteorologicalCenterPointDataset
from src.module.resnet_module import ResNetModule
from src.module.resnet_rnn_predictor import ResNetRNN_Predictor
from src.script.resnet_script import ResNetScript
from src.script.script import Script


class ResNetRNN_PredictorScript(ResNetScript):
    def create_architecture(self, datamodule: pl.LightningDataModule):
        example_input_array, _ = next(iter(datamodule.train_dataloader()))
        model_hyperparams = self.service.model_hyperparams if hasattr(self.service, 'model_hyperparams') else {}
        model = ResNetRNN_Predictor(self.service, example_input_array=example_input_array, **model_hyperparams)
        summary(model, input_size=example_input_array.shape[1:], device=str(model.device))
        return model