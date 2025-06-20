"""
Non-linear bilinear connect network as in [1].

References
----------
[1] Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang, URL: https://github.com/jnhwkim/ban-vqa
"""

from __future__ import print_function

from typing import List, Optional

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from vqa.fc import FCNet


class BCNet(nn.Module):
    """Simple class for non-linear bilinear connect network."""

    def __init__(
        self,
        v_dim: int,
        q_dim: int,
        h_dim: int,
        h_out: Optional[int],
        act: str = "ReLU",
        dropout: Optional[List[float]] = None,
        k: int = 3,
    ) -> None:
        """Initialize the network."""
        super(BCNet, self).__init__()

        if dropout is None:
            dropout = [0.2, 0.5]
        self.c = 32
        self.k = k
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.h_dim = h_dim
        self.h_out = h_out

        self.v_net = FCNet([v_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.q_net = FCNet([q_dim, h_dim * self.k], act=act, dropout=dropout[0])
        self.dropout = nn.Dropout(dropout[1])  # attention
        if k > 1:
            self.p_net = nn.AvgPool1d(self.k, stride=self.k)

        if h_out is None:
            pass
        elif h_out <= self.c:
            self.h_mat = nn.Parameter(
                torch.Tensor(1, h_out, 1, h_dim * self.k).normal_()
            )
            self.h_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        else:
            self.h_net = weight_norm(nn.Linear(h_dim * self.k, h_out), dim=None)

    def forward(self, v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.h_out is None:
            v_ = self.v_net(v).transpose(1, 2).unsqueeze(3)
            q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
            d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
            return d_.transpose(1, 2).transpose(2, 3)  # b x v x q x h_dim

        # broadcast Hadamard product, matrix-matrix production
        # fast computation but memory inefficient
        # epoch 1, time: 157.84
        if self.h_out <= self.c:
            v_ = self.dropout(self.v_net(v)).unsqueeze(1)
            q_ = self.q_net(q)
            h_ = v_ * self.h_mat  # broadcast, b x h_out x v x h_dim
            logits = torch.matmul(
                h_, q_.unsqueeze(1).transpose(2, 3)
            )  # b x h_out x v x q
            return logits + self.h_bias  # b x h_out x v x q

        # batch outer product, linear projection
        # memory efficient but slow computation
        # epoch 1, time: 304.87
        v_ = self.dropout(self.v_net(v)).transpose(1, 2).unsqueeze(3)
        q_ = self.q_net(q).transpose(1, 2).unsqueeze(2)
        d_ = torch.matmul(v_, q_)  # b x h_dim x v x q
        logits = self.h_net(d_.transpose(1, 2).transpose(2, 3))  # b x v x q x h_out
        return logits.transpose(2, 3).transpose(1, 2)  # b x h_out x v x q

    def forward_with_weights(
        self, v: torch.Tensor, q: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with given weights."""
        v_ = self.v_net(v).transpose(1, 2).unsqueeze(2)  # b x d x 1 x v
        q_ = self.q_net(q).transpose(1, 2).unsqueeze(3)  # b x d x q x 1
        logits = torch.matmul(
            torch.matmul(v_.float(), w.unsqueeze(1).float()), q_.float()
        ).type_as(v_)  # b x d x 1 x 1

        logits = logits.squeeze(3).squeeze(2)
        if self.k > 1:
            logits = logits.unsqueeze(1)  # b x 1 x d
            logits = self.p_net(logits).squeeze(1) * self.k  # sum-pooling
        return logits
