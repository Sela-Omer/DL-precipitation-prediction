from abc import ABC, abstractmethod
from typing import Callable, List, Dict

import torch
from lightning.pytorch.loggers import TensorBoardLogger

from src.helper.param_helper import convert_param_to_type


class Service(ABC):
    def __init__(self, config):
        self.config = config
        self.model_name = f"{self.config['APP']['ARCH']}"
        self.batch_size = convert_param_to_type(self.config['APP']['BATCH_SIZE'])
        self.cpu_workers = convert_param_to_type(self.config['APP']['CPU_WORKERS'])
        self.dataset_size_percent = convert_param_to_type(self.config['APP']['DATA_SUBSET_SIZE_PERCENT'])

        self.data_parameters = convert_param_to_type(self.config['DATA']['PARAMETERS'])
        self.data_years = convert_param_to_type(self.config['DATA']['YEARS'])
        self.data_cache = convert_param_to_type(self.config['DATA']['CACHE'])
        self.val_ratio = convert_param_to_type(self.config['DATA']['VAL_RATIO'])

        self.memo = {}

    @property
    @abstractmethod
    def scripts(self) -> Dict[str, Callable]:
        """
        Returns a dictionary of scripts.
        The dictionary contains a single key-value pair.
        The key is the name of the script, and the value is a function that implements the script.
        Returns:
            Dict[str, Callable]: A dictionary of scripts.
        """
        pass
