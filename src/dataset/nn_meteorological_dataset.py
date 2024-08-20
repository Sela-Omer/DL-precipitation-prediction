import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.dataset.meterorological_dataset import MeteorologicalDataset


class NN_MeteorologicalCenterPointDataset(MeteorologicalDataset):
    def _get_item(self, idx):
        """
        Get the data for a specific index.
        :param idx: The index of the data to get.
        :return: The data for the given index.
        """
        index_file = self.index_files[idx]
        data_list = self._load_data(index_file)
        center_point_data = [data[:, data.shape[1] // 2, data.shape[2] // 2] for data in data_list]
        concatenated_data = np.stack(center_point_data, axis=0)  # Shape: NUMBER_OF_PARAMETERS x TIMES
        return torch.tensor(concatenated_data, dtype=torch.float32)

    def _plot_sample(self, fig, axes, data_tensor, grouped_params):
        """
        Plot a sample of the dataset.
        :param fig: The figure to plot on.
        :param axes: The axes to plot on.
        :param data_tensor: The data tensor to plot.
        :param grouped_params: The grouped parameters to plot.
        :return:
        """
        for i, (base_param, params) in enumerate(grouped_params.items()):
            for param in params:
                param_idx = self.service.data_parameters.index(param)
                param_data = data_tensor[param_idx].numpy()

                if len(param_data.shape) == 1:  # Shape: TIMES
                    axes[i, 0].plot(param_data, label=param)
                    axes[i, 0].set_title(f"{base_param} (Time Series)")
                    axes[i, 0].set_xlabel("Time")
                    axes[i, 0].set_ylabel("Value")
                    axes[i, 1].axis('off')

            axes[i, 0].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"media/nn_{time.time_ns()}.png")
        plt.show()
