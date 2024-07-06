import torch

from src.module.resnet_module import ResNetModule
from src.module.rnn_predictor import RNNPredictor
from src.module.skip_connection_module import SkipConnectionModule
from src.service.service import Service


class ResNetRNN_Predictor(ResNetModule):
    """
    A ResNet model with RNN predictor.

    """

    def __init__(self, service: Service, *args, **kwargs):
        hidden_size = 64
        super().__init__(service, *args, override_last_lin_planes=hidden_size * 5 + 1 + hidden_size * hidden_size,
                         **kwargs)

        rnn_predictor = RNNPredictor(hidden_size)

        self.resnet34_model = SkipConnectionModule(self.resnet34_model, rnn_predictor, self._select_skip_connection,
                                                   None)

    def _select_skip_connection(self, x):
        batch_size, num_features, num_timesteps, height, width = x.shape[0], len(
            self.input_parameters), self.lookback_range + 1, x.shape[-2], x.shape[-1]
        x = x[..., height // 2, width // 2]
        x = x.reshape(batch_size, num_features, num_timesteps)
        x = torch.index_select(x, 1, torch.tensor(self.target_parameter_indices).to(x.device))
        x = x.reshape(batch_size, -1)
        return x
