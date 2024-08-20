from typing import Callable


class SumTfm(Callable):
    def __init__(self, service, sum_parameter):
        self.tfm_name = f'sum_{sum_parameter}'
        self.service = service
        self.sum_parameter = f'{sum_parameter}#sum'
        self.data_parameters = service.data_parameters.copy()

    def __call__(self, x):
        # PARAMS x TIMES x HEIGHT x WIDTH
        assert len(x.shape) == 4, f"shape of x must be (PARAMS x TIMES x HEIGHT x WIDTH) instead got {x.shape}"
        param_i = self.service.get_parameter_index(self.sum_parameter)

        x[param_i] = x[param_i].sum(axis=(-2, -1))[..., None, None].repeat((1, 1, x.shape[-2], x.shape[-1]))

        return x
