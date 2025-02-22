from abc import ABC, abstractmethod
from typing import Callable, List, Dict

import torch
from lightning.pytorch.loggers import TensorBoardLogger

from src.helper.param_helper import convert_param_to_type, convert_param_to_list
from src.transform.circular_unfold_tfm import CircularUnfoldTfm
from src.transform.correct_intensity_tfm import CorrectIntensityTfm
from src.transform.crop_time_tfm import CropTimeTfm
from src.transform.delta_suffix_tfm import DeltaSuffixTfm
from src.transform.filter_nan_tfm import FilterNanTfm
from src.transform.haversine_distance_tfm import HaversineDistanceTfm
from src.transform.norm_tfm import NormalizeTfm
from src.transform.roll_time_suffix_tfm import RollTimeSuffixTfm
from src.transform.sum_tfm import SumTfm


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
        self.environment = convert_param_to_type(self.config['APP']['ENVIRONMENT'])

        self.land_sea_mask_path = self.config['DATA']['LAND_SEA_MASK_PATH']
        self.data_parameters = convert_param_to_list(self.config['DATA']['PARAMETERS'])
        self.norm_parameters = convert_param_to_list(self.config['DATA']['NORM_PARAMETERS'])
        self.target_parameters = convert_param_to_list(self.config['DATA']['TARGET_PARAMETERS'])
        self.input_parameters = convert_param_to_list(self.config['DATA']['INPUT_PARAMETERS'])
        self.fix_times_mismatch_in_data = convert_param_to_type(self.config['DATA']['FIX_TIMES_MISMATCH_IN_DATA'])

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
        self.apply_tfms = convert_param_to_list(self.config['DATA']['APPLY_TFMS'])

        self.memo = {}
        tfm_lst = [NormalizeTfm(self),
                   CorrectIntensityTfm(self),
                   CircularUnfoldTfm(self, 'date', 1, 366),
                   CircularUnfoldTfm(self, 'lon', 0, 360),
                   SumTfm(self, 'tp'),
                   RollTimeSuffixTfm(self, -1, '-6h'),
                   CropTimeTfm(self, 0, -1),
                   DeltaSuffixTfm(self, '-6h'),
                   DeltaSuffixTfm(self, '-12h'),
                   DeltaSuffixTfm(self, '-18h'),
                   DeltaSuffixTfm(self, '-24h'),
                   FilterNanTfm(self),
                   HaversineDistanceTfm(self, drop_first_time=True),
                   HaversineDistanceTfm(self, drop_first_time=False),
                   ]
        for tfm in tfm_lst:
            assert isinstance(tfm, Callable), f"Transform {tfm} is not callable."
            assert hasattr(tfm, 'tfm_name'), f"Transform {tfm} does not have a tfm_name attribute."
        tfm_dict = {tfm.tfm_name: tfm for tfm in tfm_lst}
        self.tfms = [tfm_dict[tfm_name] for tfm_name in self.apply_tfms]

    def get_config_params(self, section: str, parameter: str, default=None):
        """
        Get the configuration parameters.
        :param section: The section of the configuration file.
        :param parameter: The parameter to get.
        :param default: The default value to return if the parameter is not found.
        :return: The value of the parameter.
        """
        return self.config.get(section, parameter, fallback=default)

    def add_tfm(self, tfm: Callable):
        """
        Adds a transform to the service.
        :param tfm: The transform to add.
        :return:
        """
        self.tfms.append(tfm)

    def apply_tfms_on_item(self, item):
        """
        Apply the transforms on the input item.
        :param item: The input item.
        :return: The transformed item.
        """
        for tfm in self.tfms:
            item = tfm(item)
        return item

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

    def remove_parameter_index(self, index: int):
        """
        Removes the parameter at the specified index.
        :param index: The index to remove.
        :return:
        """
        self.data_parameters.pop(index)

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
