"""Base model class with common functionality."""

import logging
from typing import Dict, Any
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    """Base class for all models."""

    def __init__(self, name: str = "BaseModel"):
        """Initialize base model.
        
        Args:
            name: Name of the model
        """
        super().__init__()
        self.name = name
        self._is_training = True

    def train(self, mode: bool = True) -> "BaseModel":
        """Set model to training mode.
        
        Args:
            mode: Whether to set training mode
            
        Returns:
            Self for chaining
        """
        super().train(mode)
        self._is_training = mode
        return self

    def eval(self) -> "BaseModel":
        """Set model to evaluation mode.
        
        Returns:
            Self for chaining
        """
        super().eval()
        self._is_training = False
        return self

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "non_trainable": total - trainable,
        }

    def get_model_size(self) -> float:
        """Get model size in MB.
        
        Returns:
            Model size in MB
        """
        total_params = sum(p.numel() for p in self.parameters())
        return (total_params * 4) / (1024 ** 2)

    def summary(self) -> str:
        """Get model summary as string.
        
        Returns:
            Model summary
        """
        params = self.count_parameters()
        size = self.get_model_size()
        summary = (
            f"\n{'='*60}\n"
            f"Model: {self.name}\n"
            f"{'='*60}\n"
            f"Architecture:\n{self}\n"
            f"{'='*60}\n"
            f"Parameters:\n"
            f"  Total: {params['total']:,}\n"
            f"  Trainable: {params['trainable']:,}\n"
            f"  Non-trainable: {params['non_trainable']:,}\n"
            f"Model Size: ~{size:.2f} MB\n"
            f"{'='*60}\n"
        )
        return summary

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_name": self.name,
        }
        torch.save(checkpoint, path)
        logger.info(f"Model checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location="cpu")
        self.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Model checkpoint loaded from {path}")
