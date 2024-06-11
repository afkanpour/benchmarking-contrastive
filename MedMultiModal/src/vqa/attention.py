"""
Bilinear Attention as fusion mechanism [1].

References
----------
[1] Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang, URL: https://github.com/jnhwkim/ban-vqa
"""

from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from vqa.bc import BCNet


# Bilinear Attention
class BiAttention(nn.Module):
    """Bilinear Attention Network."""

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        z_dim: int,
        glimpse: int,
        dropout: Optional[List[float]] = None,
    ) -> None:
        """Initialize the module."""
        super(BiAttention, self).__init__()

        if dropout is None:
            dropout = [0.2, 0.5]
        self.glimpse = glimpse
        self.logits = weight_norm(
            BCNet(x_dim, y_dim, z_dim, glimpse, dropout=dropout, k=3),
            name="h_mat",
            dim=None,
        )

    def forward(
        self,
        v: torch.Tensor,
        q: torch.Tensor,
        v_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        p, logits = self.forward_all(v, q, v_mask)
        return p, logits

    def forward_all(
        self,
        v: torch.Tensor,
        q: torch.Tensor,
        v_mask: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        v_num = v.size(1)
        q_num = q.size(1)
        logits = self.logits(v, q)  # b x g x v x q

        if v_mask:
            mask = (v.abs().sum(2) == 0).unsqueeze(1).unsqueeze(3).expand(logits.size())
            logits.data.masked_fill_(mask.data, -float("inf"))

        p = nn.functional.softmax(logits.view(-1, self.glimpse, v_num * q_num), 2)
        return p.view(-1, self.glimpse, v_num, q_num), logits
