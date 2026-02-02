"""Models module for neural network architectures."""

from .cnn import UrbanSceneCNN
from .base import BaseModel

__all__ = [
    "UrbanSceneCNN",
    "BaseModel",
]
