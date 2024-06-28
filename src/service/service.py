from abc import ABC, abstractmethod
from typing import Callable, List, Dict

import torch
from lightning.pytorch.loggers import TensorBoardLogger

from src.helper.param_helper import convert_param_to_type


class Service(ABC):
    """
    A class representing a service.
    :arg config: The configuration file for the service.
    
    """

    def __init__(self, config):
        self.config = config
        override_config_file = self.config['APP']['OVERWRITE_CONFIG_PATH']
        override_config_filename = override_config_file.split('/')[-1].split('.')[0]
        self.model_name = f"{self.config['APP']['ARCH']}-{override_config_filename}"
        self.batch_size = convert_param_to_type(self.config['APP']['BATCH_SIZE'])
        self.cpu_workers = convert_param_to_type(self.config['APP']['CPU_WORKERS'])

        self.data_parameters = convert_param_to_type(self.config['DATA']['PARAMETERS'])
        self.target_parameters = convert_param_to_type(self.config['DATA']['TARGET_PARAMETERS'])
        self.input_parameters = convert_param_to_type(self.config['DATA']['INPUT_PARAMETERS'])

        if len(self.target_parameters) > 0 and self.target_parameters[-1] == '':
            self.target_parameters = self.target_parameters[:-1]
        if len(self.input_parameters) > 0 and self.input_parameters[-1] == '':
            self.input_parameters = self.input_parameters[:-1]
        if len(self.data_parameters) > 0 and self.data_parameters[-1] == '':
            self.data_parameters = self.data_parameters[:-1]

        self.data_years = convert_param_to_type(self.config['DATA']['YEARS'])
        self.data_cache = convert_param_to_type(self.config['DATA']['CACHE'])
        self.val_ratio = convert_param_to_type(self.config['DATA']['VAL_RATIO'])

        self.lookback_range = convert_param_to_type(self.config['DATA']['LOOKBACK_RANGE'])
        self.forecast_range = convert_param_to_type(self.config['DATA']['FORECAST_RANGE'])

        self.network_depth = convert_param_to_type(self.config['APP']['NETWORK_DEPTH'])

        self.memo = {}

    def set_parameter_index(self, parameter: str, index: int):
        """
        Sets the index of the parameter in the data parameters.
        :param parameter: The parameter to set.
        :param index: The index to set.
        :return:
        """
        self.data_parameters[index] = parameter

    def add_parameter_at_index(self, parameter: str, index: int):
        """
        Adds a parameter at the specified index.
        :param parameter: The parameter to add.
        :param index: The index at which to add the parameter.
        :return:
        """
        self.data_parameters.insert(index, parameter)

    def get_parameter_index(self, parameter: str) -> int:
        """
        Returns the index of the parameter in the data parameters.
        :param parameter: The parameter to find.
        :return: The index of the parameter.
        """
        return self.data_parameters.index(parameter)

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
