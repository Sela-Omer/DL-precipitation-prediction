from torch import nn


class HalfBatchNorm2d(nn.BatchNorm2d):
    """
    A batch normalization module that only normalizes the first half of the features.

    """
    def __init__(self, num_features: int, *args, **kwargs) -> None:
        super().__init__(num_features // 2, *args, **kwargs)
        self.half_ind = num_features // 2

    def forward(self, x):
        # Normalize the first half of the features.
        x[:, :self.half_ind] = super().forward(x[:, :self.half_ind])
        return x
