from typing import Callable

import torch


class CorrectIntensityTfm(Callable):
    tfm_name = 'correct_intensity'

    def __init__(self, service):
        self.service = service

    def __call__(self, x):
        """
        Correct the intensity values in the input data.
        :param x:
        :return:
        """
        intensity_index = self.service.get_parameter_index('intensity')
        x[intensity_index, x[intensity_index] > 150] /= 100
        return x
