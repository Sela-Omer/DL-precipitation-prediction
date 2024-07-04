from torch import nn


class EmptyNorm(nn.Module):
    """
    An empty normalization module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the model.
        :param x: The input data.
        :return: The output of the model.
        """
        return x
