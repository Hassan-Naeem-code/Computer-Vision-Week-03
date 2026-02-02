"""Visualization utilities."""

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging

logger = logging.getLogger(__name__)


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    val_accuracies: List[float],
    output_path: str = "training_history.png",
) -> None:
    """Plot training history.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        val_accuracies: Validation accuracies per epoch
        output_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(train_losses) + 1)

    # Loss plot
    ax1.plot(epochs, train_losses, "b-o", label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-s", label="Validation Loss", linewidth=2)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, val_accuracies, "g-^", label="Validation Accuracy", linewidth=2)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Training history saved to {output_path}")
    plt.close()


def plot_results(
    test_accuracy: float, output_path: str = "test_accuracy.png"
) -> None:
    """Plot test results.
    
    Args:
        test_accuracy: Test set accuracy
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    bars = ax.bar(["Test Accuracy"], [test_accuracy], color="#2ecc71", width=0.5)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("CNN Model Performance on Test Set", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.4f}\n({height*100:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Test results saved to {output_path}")
    plt.close()


def plot_confusion_matrix(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    class_names: List[str],
    output_path: str = "confusion_matrix.png",
) -> None:
    """Plot confusion matrix.
    
    Args:
        predictions: Predicted labels
        true_labels: True labels
        class_names: List of class names
        output_path: Path to save figure
    """
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Confusion matrix saved to {output_path}")
    plt.close()


def plot_sample_images(
    images: List[np.ndarray],
    labels: List[str],
    predictions: List[str] = None,
    output_path: str = "sample_predictions.png",
) -> None:
    """Plot sample images with labels.
    
    Args:
        images: List of images to plot
        labels: List of true labels
        predictions: List of predicted labels (optional)
        output_path: Path to save figure
    """
    num_samples = len(images)
    cols = 3
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for idx, (img, label) in enumerate(zip(images, labels)):
        ax = axes[idx]
        ax.imshow(img)
        if predictions and idx < len(predictions):
            title = f"True: {label}\nPred: {predictions[idx]}"
        else:
            title = f"Label: {label}"
        ax.set_title(title, fontweight="bold")
        ax.axis("off")

    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Sample images saved to {output_path}")
    plt.close()
