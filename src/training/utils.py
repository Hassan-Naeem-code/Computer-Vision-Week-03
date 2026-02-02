"""Training utilities."""

import logging
import random
from typing import Literal
import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def get_device(device_type: Literal["auto", "cuda", "cpu"] = "auto") -> torch.device:
    """Get device (GPU or CPU).
    
    Args:
        device_type: Type of device ("auto", "cuda", or "cpu")
        
    Returns:
        torch.device object
    """
    if device_type == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_type)

    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    return device


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, value: float) -> bool:
        """Check if training should stop.
        
        Args:
            value: Current metric value (e.g., validation loss)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = value
        elif value < self.best_value + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_value = value
            self.counter = 0

        return self.early_stop

    def reset(self) -> None:
        """Reset early stopping."""
        self.counter = 0
        self.best_value = None
        self.early_stop = False
