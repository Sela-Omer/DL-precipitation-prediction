from torch import nn


class ResidualNNBlock(nn.Module):
    """
    A residual neural network block.
    :param in_features: The number of input features.
    :param out_features: The number of output features.

    """

    def __init__(self, in_features, out_features, dropout_pct=None):
        super().__init__()
        self.lin1 = nn.Linear(in_features, out_features)
        # self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(out_features, out_features)
        # self.bn2 = nn.BatchNorm1d(out_features)
        self.downsample = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_pct) if dropout_pct is not None else None

    def forward(self, x):
        identity = x

        x = self.lin1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        x = self.lin2(x)
        # x = self.bn2(x)

        x += self.downsample(identity)

        x = self.relu(x)

        if self.dropout is not None:
            x = self.dropout(x)
        return x
