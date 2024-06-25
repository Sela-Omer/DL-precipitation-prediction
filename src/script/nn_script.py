from abc import ABC

import lightning as pl

from src.datamodule.meteorological_data_module import MeteorologicalDataModule
from src.dataset.nn_meteorological_dataset import NN_MeteorologicalCenterPointDataset
from src.script.script import Script


class NNScript(Script, ABC):

    def create_architecture(self, datamodule: pl.LightningDataModule):
        pass

    def create_datamodule(self):
        data_dir = self.service.config['DATA']['PATH']
        dataset_class = NN_MeteorologicalCenterPointDataset

        datamodule = MeteorologicalDataModule(service=self.service, dataset_cls=dataset_class, data_dir=data_dir)

        if self.service.config['APP']['ENVIRONMENT'] == 'DEVELOPMENT':
            datamodule.prepare_data()
            datamodule.setup(stage='fit')
            datamodule.train_dataset.__repr__()

        return datamodule
