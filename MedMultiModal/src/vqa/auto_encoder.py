"""
Auto-encoder module for MEVF model.

This code is written by Binh X. Nguyen and Binh D. Nguyen [1].

References
----------
[1] "Mixture of Enhanced Visual Features",
    URL: https://github.com/aioz-ai/MICCAI19-MedVQA
"""

import torch
from torch import nn
from torch.nn.functional import relu


class AutoEncoderModel(nn.Module):
    """Auto-encoder module for MEVF model."""

    def __init__(self) -> None:
        """Initialize the auto-encoder."""
        super(AutoEncoderModel, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, padding=1, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 32, padding=1, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 16, padding=1, kernel_size=3)

        # Decoder
        self.tran_conv1 = nn.ConvTranspose2d(
            16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.tran_conv2 = nn.ConvTranspose2d(
            32, 64, kernel_size=3, stride=2, padding=1, output_padding=1
        )
        self.conv5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        output = relu(self.conv1(x))
        output = self.max_pool1(output)
        output = relu(self.conv2(output))
        output = self.max_pool2(output)
        return relu(self.conv3(output))

    def reconstruct_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct pass."""
        output = relu(self.tran_conv1(x))
        output = relu(self.conv4(output))
        output = relu(self.tran_conv2(output))
        return torch.sigmoid(self.conv5(output))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward and reconstruct pass."""
        output = self.forward_pass(x)
        return self.reconstruct_pass(output)
