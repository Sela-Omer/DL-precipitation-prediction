from typing import Callable

import torch


class FilterNanTfm(Callable):
    def __init__(self, service):
        self.tfm_name = f'filter_nan'
        self.service = service

    def __call__(self, x):
        # count number of replaced values nan,infinity,-infinity
        count = torch.sum(torch.isnan(x) | torch.isinf(x) | torch.isneginf(x)).item()

        if count > 0:
            print(f"Found: ({count/x.numel()} / 1) nan, inf, or -inf values in the data. Replacing them with 0.")

            # replace nan values with 0
            x[torch.isnan(x)] = 0
            # replace inf values with 0
            x[torch.isinf(x)] = 0
            # replace -inf values with 0
            x[torch.isneginf(x)] = 0



        return x
