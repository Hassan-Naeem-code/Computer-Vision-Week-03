"""CNN model for urban scene classification."""

import logging
from typing import List
import torch
import torch.nn as nn

from .base import BaseModel

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Reusable convolutional block with batch normalization and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        dropout_rate: float = 0.25,
        use_batch_norm: bool = True,
    ):
        """Initialize convolutional block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolutional kernel
            padding: Padding for convolution
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )

        if use_batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class UrbanSceneCNN(BaseModel):
    """Convolutional Neural Network for Urban Scene Classification.
    
    Architecture:
        - 3 convolutional blocks (32->64->128 filters)
        - Batch normalization after each conv layer
        - Dropout for regularization
        - 2 fully connected layers
    """

    def __init__(
        self,
        num_classes: int,
        conv_filters: List[int] = None,
        fc_hidden: int = 512,
        dropout_conv: float = 0.25,
        dropout_fc: float = 0.5,
        use_batch_norm: bool = True,
    ):
        """Initialize Urban Scene CNN.
        
        Args:
            num_classes: Number of output classes
            conv_filters: List of filters for each conv block
            fc_hidden: Number of hidden units in FC layer
            dropout_conv: Dropout rate for conv layers
            dropout_fc: Dropout rate for FC layers
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__(name="UrbanSceneCNN")

        if conv_filters is None:
            conv_filters = [32, 64, 128]

        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.fc_hidden = fc_hidden

        # Convolutional blocks
        self.conv_blocks = nn.ModuleList()
        in_channels = 3

        for out_channels in conv_filters:
            self.conv_blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout_rate=dropout_conv,
                    use_batch_norm=use_batch_norm,
                )
            )
            in_channels = out_channels

        # Calculate flattened size after convolutions
        # Input: (3, 128, 128)
        # After 3 pooling layers: (last_filter, 16, 16)
        self.flatten_size = conv_filters[-1] * 16 * 16

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, fc_hidden)
        self.bn_fc = nn.BatchNorm1d(fc_hidden) if use_batch_norm else None
        self.relu_fc = nn.ReLU()
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

        # Initialize weights
        self._init_weights()

        logger.info(f"Initialized {self.name} with {num_classes} classes")

    def _init_weights(self) -> None:
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 128, 128)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Convolutional blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        if self.bn_fc is not None:
            x = self.bn_fc(x)
        x = self.relu_fc(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)

        return x

    def get_layer_info(self) -> str:
        """Get detailed layer information.
        
        Returns:
            String with layer details
        """
        info = f"\nLayer Information:\n"
        info += f"{'Layer':<20} {'Output Shape':<20} {'Parameters':<15}\n"
        info += "-" * 55 + "\n"

        # Simulate forward pass to get layer shapes
        dummy_input = torch.randn(1, 3, 128, 128)
        x = dummy_input

        for i, conv_block in enumerate(self.conv_blocks):
            x = conv_block(x)
            params = sum(p.numel() for p in conv_block.parameters())
            info += f"ConvBlock {i+1:<13} {str(tuple(x.shape)):<20} {params:<15,}\n"

        x = x.view(x.size(0), -1)
        params = sum(self.fc1.parameters())
        info += f"FC1 {str(tuple(x.shape)):<15} {params:<15,}\n"

        x = self.fc2(x)
        params = sum(self.fc2.parameters())
        info += f"FC2 {str(tuple(x.shape)):<15} {params:<15,}\n"

        return info
