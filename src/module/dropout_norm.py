from torch import nn


class DropoutNorm(nn.Dropout2d):
    """
    A dropout module that fits in the place where norm layers are usually placed.

    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(p=0.075, inplace=False)
