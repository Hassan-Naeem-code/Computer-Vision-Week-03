"""Main entry point for training the model."""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.config import get_config
from src.data import load_dataset
from src.models import UrbanSceneCNN
from src.training import Trainer, set_seed, get_device
from src.training.utils import EarlyStopping
from src.utils import setup_logger, plot_training_history, plot_results
from src.utils.visualization import plot_confusion_matrix

# Setup root logger
root_logger = setup_logger(
    name="urban_scene_cnn",
    log_level="INFO",
)
logger = logging.getLogger(__name__)


def train_model(config_path: str = None) -> None:
    """Train the CNN model.
    
    Args:
        config_path: Path to configuration file
    """
    logger.info("=" * 70)
    logger.info("Urban Scene CNN - Training Pipeline")
    logger.info("=" * 70)

    # Load configuration
    config = get_config(config_path)
    logger.info(f"\nConfiguration:\n{config}")

    # Set random seed
    set_seed(config.random_seed)

    # Get device
    device = get_device(config.device.type)

    # Load dataset
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Loading Dataset")
    logger.info("=" * 70)
    
    train_loader, val_loader, test_loader, dataset = load_dataset(
        dataset_path=config.dataset.path,
        image_size=config.dataset.image_size,
        mean=config.dataset.mean,
        std=config.dataset.std,
        batch_size=config.training.batch_size,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        random_seed=config.random_seed,
    )

    # Initialize model
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Initializing Model")
    logger.info("=" * 70)
    
    model = UrbanSceneCNN(
        num_classes=len(dataset.classes),
        conv_filters=config.model.conv_filters,
        fc_hidden=config.model.fc_hidden,
        dropout_conv=config.model.dropout_conv,
        dropout_fc=config.model.dropout_fc,
        use_batch_norm=config.model.use_batch_norm,
    )
    
    logger.info(model.summary())

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config.training.learning_rate
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=config.training.checkpoint_dir,
    )

    # Setup early stopping (optional)
    early_stopping = None
    if config.training.early_stopping:
        early_stopping = EarlyStopping(
            patience=config.training.patience, min_delta=1e-4
        )

    # Train model
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Training Model")
    logger.info("=" * 70)
    
    history = trainer.fit(
        epochs=config.training.epochs,
        early_stopping=early_stopping,
        save_frequency=config.training.save_frequency,
    )

    # Evaluate on test set
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Evaluating Model")
    logger.info("=" * 70)

    # Load best model
    best_model_path = Path(config.training.checkpoint_dir) / "best_model.pth"
    if best_model_path.exists():
        trainer.load_checkpoint(str(best_model_path))

    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_accuracy = test_correct / test_total

    logger.info(f"\nTest Results:")
    logger.info(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    logger.info(f"  Correct predictions: {test_correct}/{test_total}")

    # Save visualizations
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Generating Visualizations")
    logger.info("=" * 70)

    Path(config.logging.output_dir).mkdir(parents=True, exist_ok=True)

    plot_training_history(
        history["train_losses"],
        history["val_losses"],
        history["val_accuracies"],
        output_path=str(
            Path(config.logging.output_dir) / "training_history.png"
        ),
    )

    plot_results(
        test_accuracy,
        output_path=str(
            Path(config.logging.output_dir) / "test_accuracy.png"
        ),
    )

    plot_confusion_matrix(
        all_predictions,
        all_labels,
        dataset.classes,
        output_path=str(
            Path(config.logging.output_dir) / "confusion_matrix.png"
        ),
    )

    logger.info("\n" + "=" * 70)
    logger.info("✅ Training Complete!")
    logger.info("=" * 70)
    logger.info(f"\nOutput files saved to: {config.logging.output_dir}")
    logger.info(f"Model checkpoints saved to: {config.training.checkpoint_dir}")
    logger.info("\n🎉 Ready for submission!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Urban Scene CNN"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    try:
        train_model(args.config)
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
