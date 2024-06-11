"""
Fully connected network as in [1] which modifies [2].

References
----------
[1] Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang, URL: https://github.com/jnhwkim/ban-vqa
[2] Hengyuan Hu, URL: https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
from __future__ import print_function

from typing import List

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class FCNet(nn.Module):
    """Simple class for non-linear fully connect network."""

    def __init__(
        self,
        dims: List[int],
        act: str = "ReLU",
        dropout: float = 0,
    ) -> None:
        """Initialize the network."""
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            if act != "":
                layers.append(getattr(nn, act)())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if act != "":
            layers.append(getattr(nn, act)())

        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.main(x)
