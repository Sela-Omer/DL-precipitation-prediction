from abc import ABC

import lightning as pl
from torchsummary import summary

from src.datamodule.meteorological_data_module import MeteorologicalDataModule
from src.dataset.cnn_meteorological_dataset import CNN_MeteorologicalDataset
from src.dataset.nn_meteorological_dataset import NN_MeteorologicalCenterPointDataset
from src.module.resnet_module import ResNetModule
from src.script.script import Script


class ResNetScript(Script, ABC):
    """
    A script for training a CNN model on meteorological data.

    """

    def create_architecture(self, datamodule: pl.LightningDataModule):
        example_input_array, _ = next(iter(datamodule.train_dataloader()))
        model_hyperparams = self.service.model_hyperparams if hasattr(self.service, 'model_hyperparams') else {}
        model = ResNetModule(self.service, example_input_array=example_input_array, **model_hyperparams)
        summary(model, input_size=example_input_array.shape[1:], device=str(model.device))
        return model

    def create_datamodule(self):
        """
        Create the data module for the script.
        :return: The data module for the script.
        """
        data_dir = self.service.config['DATA']['PATH']
        dataset_class = CNN_MeteorologicalDataset

        datamodule = MeteorologicalDataModule(service=self.service, dataset_cls=dataset_class, data_dir=data_dir)
        datamodule.prepare_data()
        datamodule.setup()

        return datamodule
