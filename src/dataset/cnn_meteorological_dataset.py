import numpy as np
import torch

from src.dataset.meterorological_dataset import MeteorologicalDataset


class CNN_MeteorologicalDataset(MeteorologicalDataset):
    def __iter__(self):
        for index_file in self.index_files:
            data_list = self._load_data(index_file)
            concatenated_data = np.stack(data_list, axis=0)  # Shape: NUMBER_OF_PARAMETERS x TIMES x HEIGHT x WIDTH
            yield torch.tensor(concatenated_data, dtype=torch.float32)
