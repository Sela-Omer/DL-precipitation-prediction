import torch
from torch import nn

from src.module.cnn_module import CNNModule
from src.module.skip_connection_module import SkipConnectionModule


class SkipConnectionCNNModule(CNNModule):
    """
    :param service: A service object that contains the configuration parameters.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        body = list(self.cnn.body[:-1])
        body_out_features = self.cnn.fc.in_features
        skip_len = (self.lookback_range + 1) * len(self.target_parameters)
        last_lin = nn.Linear(body_out_features + skip_len,
                             len(self.target_parameters))
        nn.init.constant_(last_lin.weight, 0)

        # Set the last k weights to 1/k
        with torch.no_grad():
            last_lin.weight[:, -skip_len:] = 1 / skip_len
        # Initialize all biases to 0
        nn.init.constant_(last_lin.bias, 0)

        self.cnn.body = SkipConnectionModule(nn.Sequential(*body), nn.Sequential(last_lin),
                                             self._select_skip_connection,
                                             1)

    def _select_skip_connection(self, x):
        h, w = x.shape[-2:]
        x = x[..., h // 2, w // 2]
        x = x.reshape(x.shape[0], len(self.input_parameters), self.lookback_range + 1)
        x = torch.index_select(x, 1, torch.tensor(self.target_parameter_indices).to(x.device))
        for t in range(x.shape[2] - 1):
            prv = x[:, :, t]
            nxt = x[:, :, -1]
            times = x.shape[2] - t - 1
            x[:, :, t] = (nxt + ((self.forecast_range) * (nxt - prv)) / times)
        return x.reshape(x.shape[0], -1)
