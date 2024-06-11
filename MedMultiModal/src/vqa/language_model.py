"""
Bilinear Attention as fusion mechanism [1].

References
----------
[1] Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang, URL: https://github.com/jnhwkim/ban-vqa
"""
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class WordEmbedding(nn.Module):
    """Word Embedding.

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """

    def __init__(self, ntoken: int, emb_dim: int, dropout: float, op: str = "") -> None:
        """Initialize the module."""
        super(WordEmbedding, self).__init__()
        self.op = op
        self.emb = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
        if "c" in op:
            self.emb_ = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
            self.emb_.weight.requires_grad = False  # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(
        self,
        np_file: str,
        tfidf: Optional[torch.Tensor] = None,
        tfidf_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize weights."""
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[: self.ntoken] = weight_init
        if tfidf is not None:
            if tfidf_weights is not None and tfidf_weights.size > 0:
                weight_init = torch.cat(
                    [weight_init, torch.from_numpy(tfidf_weights)], 0
                )
            weight_init = tfidf.matmul(weight_init)  # (N x N') x (N', F)
            self.emb_.weight.requires_grad = True
        if "c" in self.op:
            self.emb_.weight.data[: self.ntoken] = weight_init.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        emb = self.emb(x)
        if "c" in self.op:
            emb = torch.cat((emb, self.emb_(x)), 2)
        return self.dropout(emb)


class QuestionEmbedding(nn.Module):
    """Module for question embedding."""

    def __init__(
        self,
        in_dim: int,
        num_hid: int,
        nlayers: int,
        bidirect: bool,
        dropout: float,
        rnn_type: str = "GRU",
    ) -> None:
        """Initialize the module."""
        super(QuestionEmbedding, self).__init__()
        assert rnn_type in ["LSTM", "GRU"]
        rnn_cls = getattr(nn, rnn_type) if rnn_type in ["LSTM", "GRU"] else None

        self.rnn = rnn_cls(
            in_dim,
            num_hid,
            nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True,
        )

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(
        self, batch: Dict[str, Any]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Initialize hidden layers."""
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (
            self.nlayers * self.ndirections,
            batch,
            self.num_hid // self.ndirections,
        )
        if self.rnn_type == "LSTM":
            return (
                Variable(weight.new(*hid_shape).zero_()),
                Variable(weight.new(*hid_shape).zero_()),
            )
        return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # shape of x is (batch x sequence x in_dim)
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, : self.num_hid]
        backward = output[:, 0, self.num_hid :]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of questions."""
        # shape of x is (batch x sequence x in_dim)
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        output, hidden = self.rnn(x, hidden)
        return output
