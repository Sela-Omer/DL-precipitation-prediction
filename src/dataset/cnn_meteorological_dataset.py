import numpy as np
import torch
from matplotlib import pyplot as plt

from src.dataset.meterorological_dataset import MeteorologicalDataset


class CNN_MeteorologicalDataset(MeteorologicalDataset):
    def _get_item(self, idx):
        """
        Get the data for a specific index.
        :param idx: The index of the data to get.
        :return: The data for the given index.
        """
        index_file = self.index_files[idx]
        data_list = self._load_data(index_file)
        concatenated_data = np.stack(data_list, axis=0)  # Shape: NUMBER_OF_PARAMETERS x TIMES x HEIGHT x WIDTH
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

                if len(param_data.shape) == 3:  # Shape: TIMES x HEIGHT x WIDTH
                    # Plot the average over HEIGHT and WIDTH
                    time_avg = param_data.mean(axis=(1, 2))  # Average over HEIGHT and WIDTH
                    axes[i, 0].plot(time_avg, label=param)
                    axes[i, 0].set_title(f"{base_param} (Time Avg)")
                    axes[i, 0].set_xlabel("Time")
                    axes[i, 0].set_ylabel("Avg Value")

                    # Add an imshow plot to visualize the first time slice of the first parameter
                    if param == params[0]:
                        im = axes[i, 1].imshow(param_data[0, :, :], cmap='viridis', origin='lower')
                        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)
                        axes[i, 1].set_title(f"{param} (First Time Slice)")
                    else:
                        axes[i, 1].axis('off')

                elif len(param_data.shape) == 1:  # Shape: TIMES
                    axes[i, 0].plot(param_data, label=param)
                    axes[i, 0].set_title(f"{base_param} (Time Series)")
                    axes[i, 0].set_xlabel("Time")
                    axes[i, 0].set_ylabel("Value")
                    axes[i, 1].axis('off')

            axes[i, 0].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # Additional imshow plots for multiple time steps and pressures
        num_rows = len(grouped_params)
        fig, axes = plt.subplots(num_rows, 5, figsize=(15, 4 * num_rows))
        for i, (base_param, params) in enumerate(grouped_params.items()):
            for t in range(min(5, data_tensor.shape[1])):  # Plot up to 5 time steps
                for j, param in enumerate(params):
                    param_idx = self.service.data_parameters.index(param)
                    param_data = data_tensor[param_idx].numpy()

                    if len(param_data.shape) == 3:  # Shape: TIMES x HEIGHT x WIDTH
                        ax = axes[i, t]
                        im = ax.imshow(param_data[t, :, :], cmap='viridis', origin='lower')
                        if j == 0:  # Add colorbar only once per row
                            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        ax.set_title(f"{param} (Time step {t})")

        plt.tight_layout()
        plt.show()




