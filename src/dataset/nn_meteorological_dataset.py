import numpy as np
import torch

from src.dataset.meterorological_dataset import MeteorologicalDataset


class NN_MeteorologicalCenterPointDataset(MeteorologicalDataset):
    def __iter__(self):
        for index_file in self.index_files:
            data_list = self._load_data(index_file)
            center_point_data = [data[:, data.shape[1] // 2, data.shape[2] // 2] for data in data_list]
            concatenated_data = np.stack(center_point_data, axis=0)  # Shape: NUMBER_OF_PARAMETERS x TIMES
            yield torch.tensor(concatenated_data, dtype=torch.float32)
