from typing import Callable

import torch


class stochastic_times_collate_fn(Callable):
    def __init__(self, lookback_range: int, forecast_range: int):
        super().__init__()
        self.lookback_range = lookback_range
        self.forecast_range = forecast_range

    def __call__(self, batch):
        batch_size = len(batch)
        numel_in_shape = len(batch[0].shape)

        assert numel_in_shape in [4, 2], f"Expected 4 or 2 dimensions in batch, got {numel_in_shape}"

        if numel_in_shape == 4:
            # Assuming each element in the batch is of shape: PARAMS x TIMES x HEIGHT x WIDTH
            params, _, height, width = batch[0].shape
            # Initialize an empty tensor to hold the concatenated data
            X = torch.zeros(batch_size, params, self.lookback_range + 1, height, width)
            y = torch.zeros(batch_size, params, 1, height, width)
        else:
            # Assuming each element in the batch is of shape: PARAMS x TIMES
            params, _, height, width = batch[0].shape
            # Initialize an empty tensor to hold the concatenated data
            X = torch.zeros(batch_size, params, self.lookback_range + 1)
            y = torch.zeros(batch_size, params, 1)

        for i, item in enumerate(batch):
            times = item.shape[1]
            stochastic_time = torch.randint(self.lookback_range, times - self.forecast_range, (1,)).item()
            X[i] = item[:, stochastic_time - self.lookback_range:stochastic_time + 1]
            y[i] = item[:, stochastic_time + self.forecast_range][:, None]

        return X, y
