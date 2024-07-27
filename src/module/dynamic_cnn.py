import torch
from torch import nn

from src.module.flatten_module import FlattenModule


class DynamicCNN(nn.Module):
    def __init__(self, norm_module_class, pool_module_class, in_channels, target_len, depth):
        super(DynamicCNN, self).__init__()
        self.target_len = target_len
        self.depth = depth

        layers = []
        out_channels = 32

        for _ in range(depth):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(norm_module_class(out_channels))
            layers.append(pool_module_class(kernel_size=2, stride=2))

            in_channels = out_channels
            out_channels *= 2

        self.conv_layers = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = FlattenModule(1)
        self.fc = nn.Linear(in_channels, self.target_len)
        self.body = nn.Sequential(*[self.conv_layers, self.avgpool, self.flatten, self.fc])

    def forward(self, x):
        return self.body(x)
