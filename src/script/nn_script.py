from abc import ABC

import lightning as pl

from src.datamodule.meteorological_data_module import MeteorologicalDataModule
from src.dataset.nn_meteorological_dataset import NN_MeteorologicalCenterPointDataset
from src.script.script import Script


class NNScript(Script, ABC):
    """
    A script for training a neural network model on meteorological data.

    """

    def create_architecture(self, datamodule: pl.LightningDataModule):
        pass

    def create_datamodule(self):
        """
        Create the data module for the script.
        :return: The data module for the script.
        """
        data_dir = self.service.config['DATA']['PATH']
        dataset_class = NN_MeteorologicalCenterPointDataset

        datamodule = MeteorologicalDataModule(service=self.service, dataset_cls=dataset_class, data_dir=data_dir)

        return datamodule
