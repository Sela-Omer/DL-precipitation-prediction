from abc import ABC

from src.dataset.nn_meteorological_dataset import NN_MeteorologicalCenterPointDataset
from src.script.script import Script
import lightning as pl


class DataAnalysisScript(Script, ABC):

    def create_architecture(self, datamodule: pl.LightningDataModule):
        pass

    def create_datamodule(self):
        """
        Create and return an instance of the AudioDataModule.

        This method creates an instance of the AudioDataModule using the provided service, data directory, and audio dataset class.
        If the application environment is set to 'DEVELOPMENT', it also prints the representation of the train dataset.

        Returns:
            AudioDataModule: The created instance of the AudioDataModule.
        """
        # Get the data directory and audio dataset class from the service configuration
        data_dir = self.service.config['DATA']['PATH']
        dataset_class = NN_MeteorologicalCenterPointDataset

        # Create an instance of the AudioDataModule
        datamodule = AudioDataModule(service=self.service, data_dir=data_dir, audio_dataset_class=audio_dataset_class)

        # If the application environment is set to 'DEVELOPMENT', print the representation of the train dataset
        if self.service.config['APP']['ENVIRONMENT'] == 'DEVELOPMENT':
            datamodule.prepare_data()
            datamodule.setup(stage='fit')
            datamodule.train_dataset.__repr__()

        return datamodule