import json
import os
import random
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class MeteorologicalDataset(Dataset):
    def __init__(self, data_dir, parameters, years, cache_dir='cache'):
        self.data_dir = data_dir
        self.parameters = parameters
        self.years = years
        self.cache_dir = cache_dir
        self.cache_file = self._generate_cache_filename()

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        self.index_files = self._load_or_create_cache()

    def _generate_cache_filename(self):
        cache_key = f"{'_'.join(self.parameters)}_{'_'.join(self.years)}"
        return os.path.join(self.cache_dir, f"cache_{cache_key}.json")

    def _get_index_files(self):
        index_files = []
        dropped_files_stats = {param: 0 for param in self.parameters}

        for year in self.years:
            year_dir = os.path.join(self.data_dir, self.parameters[0].split('_')[0], str(year))
            if os.path.exists(year_dir):
                for filename in os.listdir(year_dir):
                    if filename.endswith('.npy'):
                        all_params_exist = True
                        for param in self.parameters:
                            param_parts = param.split('_')
                            if len(param_parts) > 1:
                                file_path = os.path.join(self.data_dir, param_parts[0], param_parts[1], year,
                                                         filename)
                            else:
                                file_path = os.path.join(self.data_dir, param, year, filename)


                            if not os.path.exists(file_path):
                                all_params_exist = False
                                dropped_files_stats[param] += 1


                        if all_params_exist:
                            index_files.append(filename)

        print("Dropped files statistics:")
        for param, count in dropped_files_stats.items():
            print(f"{param}: {count}")

        return index_files

    def _load_or_create_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                index_files = json.load(f)
        else:
            index_files = self._get_index_files()
            with open(self.cache_file, 'w') as f:
                json.dump(index_files, f)
        return index_files

    def _load_data(self, index_file):
        data_list = []
        for param in self.parameters:
            param_parts = param.split('_')
            if len(param_parts) > 1:
                file_path = os.path.join(self.data_dir, param_parts[0], param_parts[1], index_file.split('_')[1],
                                         index_file)
            else:
                file_path = os.path.join(self.data_dir, param, index_file.split('_')[1], index_file)

            data = np.load(file_path)
            data_list.append(data)
        return data_list

    def set_index_files(self, index_files):
        self.index_files = index_files

    def __len__(self):
        return len(self.index_files)

    def __repr__(self):
        repr_str = f"MeteorologicalDataset(data_dir={self.data_dir}, parameters={self.parameters}, years={self.years}, cache_dir={self.cache_dir})\n"
        repr_str += f"Number of samples: {len(self)}\n"
        repr_str += "Sample visualizations:\n"

        random_indices = random.sample(range(len(self)), min(3, len(self)))

        grouped_params = defaultdict(list)
        for param in self.parameters:
            base_param = param.split('_')[0]
            grouped_params[base_param].append(param)

        for idx in random_indices:
            data_tensor = self[idx]
            num_rows = len(grouped_params)
            fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows), gridspec_kw={'width_ratios': [3, 1]})
            fig.suptitle(f"Sample {idx}", fontsize=16)

            self._plot_sample(fig, axes, data_tensor, grouped_params)

        return repr_str

    def _plot_sample(self, fig, axes, data_tensor, grouped_params):
        raise NotImplementedError("This method should be implemented by subclasses.")

    # def __repr__(self):
    #     repr_str = f"MeteorologicalDataset(data_dir={self.data_dir}, parameters={self.parameters}, years={self.years}, cache_dir={self.cache_dir})\n"
    #     repr_str += f"Number of samples: {len(self)}\n"
    #     repr_str += "Sample visualizations:\n"
    #
    #     random_indices = random.sample(range(len(self)), min(3, len(self)))
    #
    #     grouped_params = defaultdict(list)
    #     for param in self.parameters:
    #         base_param = param.split('_')[0]
    #         grouped_params[base_param].append(param)
    #
    #     for idx in random_indices:
    #         data_tensor = self[idx]
    #         num_rows = len(grouped_params)
    #         fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows), gridspec_kw={'width_ratios': [3, 1]})
    #         fig.suptitle(f"Sample {idx}", fontsize=16)
    #
    #         for i, (base_param, params) in enumerate(grouped_params.items()):
    #             for param in params:
    #                 param_idx = self.parameters.index(param)
    #                 param_data = data_tensor[param_idx].numpy()
    #
    #                 if len(param_data.shape) == 3:  # Shape: TIMES x HEIGHT x WIDTH
    #                     # Plot the average over HEIGHT and WIDTH
    #                     time_avg = param_data.mean(axis=(1, 2))  # Average over HEIGHT and WIDTH
    #                     axes[i, 0].plot(time_avg, label=param)
    #                     axes[i, 0].set_title(f"{base_param} (Time Avg)")
    #                     axes[i, 0].set_xlabel("Time")
    #                     axes[i, 0].set_ylabel("Avg Value")
    #
    #                     # Add an imshow plot to visualize the first time slice of the first parameter
    #                     if param == params[0]:
    #                         im = axes[i, 1].imshow(param_data[0, :, :], cmap='viridis', origin='lower')
    #                         fig.colorbar(im, ax=axes[i, 1])
    #                         axes[i, 1].set_title(f"{param} (First Time Slice)")
    #                     else:
    #                         axes[i, 1].axis('off')
    #
    #                 elif len(param_data.shape) == 1:  # Shape: TIMES
    #                     axes[i, 0].plot(param_data, label=param)
    #                     axes[i, 0].set_title(f"{base_param} (Time Series)")
    #                     axes[i, 0].set_xlabel("Time")
    #                     axes[i, 0].set_ylabel("Value")
    #                     axes[i, 1].axis('off')
    #
    #             axes[i, 0].legend()
    #
    #         plt.tight_layout(rect=[0, 0, 1, 0.96])
    #         plt.show()
    #
    #     return repr_str



