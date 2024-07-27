import torch
from torch import nn


class FlattenModule(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self,x):
        return torch.flatten(x,*self.args,**self.kwargs)
