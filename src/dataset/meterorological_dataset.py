import hashlib
import json
import os

import numpy as np
from torch.utils.data import IterableDataset


class MeteorologicalDataset(IterableDataset):
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
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"cache_{cache_hash}.json")

    def _get_index_files(self):
        index_files = []
        for year in self.years:
            year_dir = os.path.join(self.data_dir, self.parameters[0], str(year))
            if os.path.exists(year_dir):
                for filename in os.listdir(year_dir):
                    if filename.endswith('.npy'):
                        index_files.append(filename)
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
            file_path = os.path.join(self.data_dir, param, index_file.split('_')[0], index_file)
            data = np.load(file_path)
            data_list.append(data)
        return data_list


