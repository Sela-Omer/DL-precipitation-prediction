from abc import ABC
from typing import Tuple

import lightning as pl
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.helper.param_helper import convert_param_to_type
from src.script.script import Script


class DataAnalysisScript(Script, ABC):
    """
    A script for analyzing the data in a dataset.

    """

    def create_trainer(self, callbacks: list):
        pass

    def create_architecture(self, datamodule: pl.LightningDataModule):
        pass

    def generate_stats_dataloader(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate the mean and standard deviation of the data in the dataloader.
        :param dataloader: The dataloader to analyze.
        :return: The mean and standard deviation of each parameter in the data.
        """
        param_sum = None
        param_sum_squared = None
        count = 0

        # Iterate over the dataloader
        for X, y in tqdm(dataloader):
            # Generate statistics for the current batch
            param_sum_batch, param_sum_squared_batch, count_batch = self.generate_stats_one_batch(X)

            if param_sum is None:
                param_sum = param_sum_batch
                param_sum_squared = param_sum_squared_batch
            else:
                param_sum += param_sum_batch
                param_sum_squared += param_sum_squared_batch
            count += count_batch

        mean = param_sum / count
        std = np.sqrt((param_sum_squared / count) - (mean ** 2))

        return mean, std

    def generate_stats_one_batch(self, X):
        if len(X.shape) == 5:
            # BATCH x PARAMS x TIMES x HEIGHT x WIDTH
            param_sum = X.sum(axis=(0, 2, 3, 4))
            param_sum_squared = (X ** 2).sum(axis=(0, 2, 3, 4))
            return param_sum, param_sum_squared, X.shape[0] * X.shape[2] * X.shape[3] * X.shape[4]
        elif len(X.shape) == 3:
            # BATCH x PARAMS x TIMES
            param_sum = X.sum(axis=(0, 2))
            param_sum_squared = (X ** 2).sum(axis=(0, 2))
            return param_sum, param_sum_squared, X.shape[0] * X.shape[2]

        raise NotImplementedError(f"generate_stats_one_batch not implemented for shape {X.shape}")

    def __call__(self):
        """
        This method orchestrates the training process.
        It creates the data module, architecture, callbacks, and trainer,
        and then fits the model using the trainer.
        """
        # Create the data module
        datamodule = self.create_datamodule()
        datamodule.prepare_data()
        datamodule.setup(stage='fit')
        print(datamodule.train_dataset.__repr__())

        if self.service.config['DATA_ANALYSIS']['EXECUTE_MODEL_STATS_CALCULATION'] == 'True':
            train_dl = datamodule.train_dataloader()

            # Generate statistics for the audio data in the dataloader
            mean, std = self.generate_stats_dataloader(train_dl)

            # Write the statistics
            file_prefix = f'stats/{self.service.model_name}'
            torch.save(mean, f'{file_prefix}-mean.pt')
            torch.save(std, f'{file_prefix}-std.pt')
