"""Trainer class for model training."""

import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from .utils import EarlyStopping

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for training neural networks."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        checkpoint_dir: str = "./checkpoints",
    ):
        """Initialize trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimization algorithm
            device: Device (CPU/GPU)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []
        self.best_val_accuracy = 0.0

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch.
        
        Returns:
            Tuple of (train_loss, train_accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader, desc="Training", leave=False, disable=False
        )

        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy

    def validate(self) -> Tuple[float, float]:
        """Validate the model.
        
        Returns:
            Tuple of (val_loss, val_accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader, desc="Validating", leave=False, disable=False
            )

            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_accuracy = correct / total

        return epoch_loss, epoch_accuracy

    def fit(
        self,
        epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        save_frequency: int = 5,
    ) -> Dict[str, List[float]]:
        """Train the model for multiple epochs.
        
        Args:
            epochs: Number of epochs
            early_stopping: Early stopping object
            save_frequency: Save checkpoint every N epochs
            
        Returns:
            Dictionary with training history
        """
        logger.info(f"\nStarting training for {epochs} epochs...")
        logger.info(f"Device: {self.device}")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            logger.info(
                f"\nEpoch [{epoch+1}/{epochs}]"
            )
            logger.info(
                f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"
            )
            logger.info(
                f"  Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}"
            )

            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.save_checkpoint(f"{self.checkpoint_dir}/best_model.pth")
                logger.info(f"  ✓ Best model saved! (Val Acc: {self.best_val_accuracy:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % save_frequency == 0:
                self.save_checkpoint(
                    f"{self.checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth"
                )

            # Early stopping
            if early_stopping is not None:
                if early_stopping(val_loss):
                    logger.info(f"\n✓ Early stopping triggered at epoch {epoch+1}")
                    break

        logger.info("\n" + "=" * 60)
        logger.info(f"Training Complete!")
        logger.info(f"Best Validation Accuracy: {self.best_val_accuracy:.4f}")
        logger.info("=" * 60)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "epoch": len(self.train_losses),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_accuracy": self.best_val_accuracy,
        }
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.val_accuracies = checkpoint.get("val_accuracies", [])
        self.best_val_accuracy = checkpoint.get("best_val_accuracy", 0.0)
        logger.info(f"Checkpoint loaded from {path}")
