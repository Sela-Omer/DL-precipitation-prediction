import hashlib
import json
import os
import random
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import List

import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from src.dataset.meterorological_dataset import MeteorologicalDataset
from src.service.service import Service


class MultiMeteorologicalDataset(Dataset):
    def __init__(self, ds_lst: List[MeteorologicalDataset], mode='intersection'):
        super().__init__()
        assert len(ds_lst) > 0, "At least one dataset must be provided."

        self.ds_lst = ds_lst
        self.mode = mode

        assert mode in ['intersection'], f"Invalid mode: {mode}"
        if mode == 'intersection':
            index_files_set = set(self.ds_lst[0].index_files)
            for ds in self.ds_lst:
                index_files_set = index_files_set.intersection(ds.index_files)
            self.index_files = list(index_files_set)
            for ds in self.ds_lst:
                ds.index_files = self.index_files

    def __len__(self):
        return len(self.index_files)

    def __repr__(self):
        for ds in self.ds_lst:
            ds.__repr__()

    def __getitem__(self, idx):
        data_lst = [ds[idx] for ds in self.ds_lst]
        return tuple(data_lst)