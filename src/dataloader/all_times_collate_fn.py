from typing import Callable, List, Tuple

import torch


class all_times_collate_fn(Callable):
    """
    A callable class that generates samples for training a model with stochastic lookback and forecast ranges.
    :arg lookback_range: The number of time steps to look back in the past.
    :arg forecast_range: The number of time steps to forecast into the future.

    """

    def __init__(self, lookback_range: int, forecast_range: int):
        super().__init__()
        self.lookback_range = lookback_range
        self.forecast_range = forecast_range

    def __call__(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates samples for training a model with stochastic lookback and forecast ranges.
        :param batch: A list of tensors, each representing a single sample.
        :return: A tuple containing the input and output tensors.

        """
        X_list = []
        y_list = []

        for item in batch:
            times = item.shape[1]
            for t in range(self.lookback_range, times - self.forecast_range):
                X = item[:, t - self.lookback_range:t + 1]
                y = item[:, t + self.forecast_range][:, None]
                X_list.append(X)
                y_list.append(y)

        # Concatenate all generated samples along the batch dimension
        X_concatenated = torch.stack(X_list, dim=0)
        y_concatenated = torch.stack(y_list, dim=0)

        return X_concatenated, y_concatenated
