import hashlib
import json
import os
import random
from abc import abstractmethod, ABC
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from src.service.service import Service


class MeteorologicalDataset(ABC, Dataset):
    """
    A PyTorch Dataset for meteorological datasets.
    :param data_dir: The directory containing the data files.
    :param parameters: The parameters to load.
    :param years: The years to load.
    :param cache_dir: The directory to store the cache files.

    """

    def __init__(self, service: Service, data_dir, years, cache_dir='cache', num_of_random_samples_repr=3):
        self.service = service
        self.data_dir = data_dir
        self.years = years
        self.cache_dir = cache_dir
        self.cache_file = self._generate_cache_filename()

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        self.index_files = self._load_or_create_cache()
        self.num_of_random_samples_repr = num_of_random_samples_repr

    def _generate_cache_filename(self):
        """
        Generate the cache filename.
        :return:
        """
        cache_key = f"{'_'.join(self.service.data_parameters)}_{'_'.join(self.years)}"

        # Encode the cache key using MD5
        md5_hash = hashlib.md5(cache_key.encode()).hexdigest()

        return os.path.join(self.cache_dir, f"cache_{md5_hash}.json")

    def _get_index_files(self):
        """
        Get the index files for the dataset.
        :return: The index files for the dataset.
        """
        index_files = []
        dropped_files_stats = {param: 0 for param in self.service.data_parameters}

        for year in self.years:
            year_dir = os.path.join(self.data_dir, 'intensity', str(year))
            if os.path.exists(year_dir):
                for filename in os.listdir(year_dir):
                    if filename.endswith('.npy'):
                        all_params_exist = True
                        for param in self.service.data_parameters:
                            if "#" in param:
                                param = param.split("#")[0]
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

        lat_filtered_index_files = self._filter_index_files_with_lat(index_files)
        print(
            f"Number of files with |lat|<70: {len(lat_filtered_index_files)}. dropped files: {len(index_files) - len(lat_filtered_index_files)}")

        return lat_filtered_index_files

    def _filter_index_files_with_lat(self, index_files):
        """
        Filter the index files based on the latitude.
        :param index_files: The index files to filter.
        :return: The filtered index files with |lat|<70.
        """
        filtered_index_files = []
        for index_file in index_files:
            lat_file_path = os.path.join(self.data_dir, 'lat', index_file.split('_')[1], index_file)
            lat_mat = np.load(lat_file_path)
            if np.abs(lat_mat).max() < 70:
                filtered_index_files.append(index_file)
        return filtered_index_files

    def _load_or_create_cache(self):
        """
        Load the cache file if it exists, otherwise create it.
        :return: The index files for the dataset.
        """
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                index_files = json.load(f)
        else:
            index_files = self._get_index_files()
            with open(self.cache_file, 'w') as f:
                json.dump(index_files, f)
        return index_files

    def _load_data(self, index_file):
        """
        Load the data for a specific index file.
        :param index_file: The index file to load.
        :return: The data for the given index file.
        """
        data_list = []
        for param in self.service.data_parameters:
            if "#" in param:
                param = param.split("#")[0]
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
        """
        Set the index files for the dataset.
        :param index_files: The index files to set.
        :return:
        """
        self.index_files = index_files

    def __len__(self):
        return len(self.index_files)

    def __repr__(self):
        """
        Get the string representation of the dataset.
        :return: The string representation of the dataset.
        """
        repr_str = f"MeteorologicalDataset(data_dir={self.data_dir}, parameters={self.service.data_parameters}, years={self.years}, cache_dir={self.cache_dir})\n"
        repr_str += f"Number of samples: {len(self)}\n"

        random_indices = random.sample(range(len(self)), min(self.num_of_random_samples_repr, len(self)))

        for idx in random_indices:
            data_tensor = self[idx]
            grouped_params = defaultdict(list)
            for param in self.service.data_parameters:
                base_param = param.split('_')[0]
                grouped_params[base_param].append(param)
            num_rows = len(grouped_params)
            fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows), gridspec_kw={'width_ratios': [3, 1]})
            fig.suptitle(f"Sample {idx}", fontsize=16)

            self._plot_sample(fig, axes, data_tensor, grouped_params)

        return repr_str

    def __getitem__(self, idx):
        """
        Get the data for a specific index.
        :param idx: The index of the data to get.
        :return: The data for the given index.
        """
        item = self._get_item(idx)
        return self.service.apply_tfms_on_item(item)

    @abstractmethod
    def _get_item(self, idx):
        """
        Get the data for a specific index.
        :param idx: The index of the data to get.
        :return:
        """
        pass

    @abstractmethod
    def _plot_sample(self, fig, axes, data_tensor, grouped_params):
        """
        Plot a sample of the dataset.
        :param fig:
        :param axes:
        :param data_tensor:
        :param grouped_params:
        :return:
        """
        pass
