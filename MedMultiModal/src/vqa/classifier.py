"""
Bilinear Attention as fusion mechanism [1].

References
----------
[1] Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang, URL: https://github.com/jnhwkim/ban-vqa
"""

import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn.utils.weight_norm import weight_norm


class SimpleClassifier(nn.Module):
    """Simple MLP layer as classification head."""

    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        out_dim: int,
        args: DictConfig,
    ) -> None:
        """Initialize the module."""
        super(SimpleClassifier, self).__init__()
        activation_dict = {"relu": nn.ReLU()}
        try:
            activation_func = activation_dict[args.activation]
        except Exception as exc:
            raise AssertionError(args.activation + " is not supported yet!") from exc
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            activation_func,
            nn.Dropout(args.dropout),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.main(x)
