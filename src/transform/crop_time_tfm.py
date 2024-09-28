from typing import Callable


class CropTimeTfm(Callable):
    def __init__(self, service, min_time_ind: int, max_time_ind: int):
        self.tfm_name = f'crop_time_[{min_time_ind}_{max_time_ind})'
        self.service = service
        self.min_time_ind = min_time_ind
        self.max_time_ind = max_time_ind

    def __call__(self, x):
        # PARAMS x TIMES x ...

        x = x[:, self.min_time_ind:self.max_time_ind]

        return x
