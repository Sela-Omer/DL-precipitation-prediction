from typing import Callable

import torch


class NoiseTfm(Callable):
    tfm_name = 'noise'

    def __init__(self, service, noise_param_index: int, noise_mean=0, noise_std=0.5):
        self.service = service
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.noise_param_index = noise_param_index

    def set_noise_param_index(self, noise_param_index: int):
        self.noise_param_index = noise_param_index

    def __call__(self, x):
        """
        Add noise to the input data.
        :param x: The input data.
        :return: data with added noise.
        """
        # PARAMS x TIMES x HEIGHT x WIDTH
        noise = torch.normal(self.noise_mean, self.noise_std,
                             size=[1 for _ in x[self.noise_param_index].shape]).to(device=x.device)
        noise = noise.repeat(*x[self.noise_param_index].shape)

        x[self.noise_param_index] = x[self.noise_param_index] + noise
        return x
