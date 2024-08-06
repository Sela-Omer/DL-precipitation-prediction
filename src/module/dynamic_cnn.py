import torch
from torch import nn

from src.module.flatten_module import FlattenModule


class DynamicCNN(nn.Module):
    def __init__(self, norm_module_class, pool_module_class, in_channels, target_len, depth, lin_ch_mult=2):
        super(DynamicCNN, self).__init__()
        self.target_len = target_len
        self.depth = depth

        cnn_layers = []
        out_channels = 32

        for i in range(4):
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2))
            cnn_layers.append(norm_module_class(out_channels))
            cnn_layers.append(pool_module_class(kernel_size=2, stride=2))

            in_channels = out_channels
            out_channels *= 2

        self.conv_layers = nn.Sequential(*cnn_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = FlattenModule(1)

        layers = []
        for i in range(depth-4):
            layers.append(nn.Linear(in_channels,out_channels))
            layers.append(nn.ReLU())
            in_channels = out_channels
            out_channels *= lin_ch_mult

        self.layers = nn.Sequential(*layers)

        self.fc = nn.Linear(in_channels, self.target_len)
        self.body = nn.Sequential(*[self.conv_layers, self.avgpool, self.flatten, self.layers, self.fc])

    def forward(self, x):
        return self.body(x)
