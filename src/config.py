"""Configuration management for the project."""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    path: str = "./data/MIT_Places_Urban_Subset"
    image_size: int = 128
    num_classes: int = 5
    download: bool = False
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    mean: list = None
    std: list = None

    def __post_init__(self):
        if self.mean is None:
            self.mean = [0.485, 0.456, 0.406]
        if self.std is None:
            self.std = [0.229, 0.224, 0.225]


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "UrbanSceneCNN"
    conv_filters: list = None
    fc_hidden: int = 512
    dropout_conv: float = 0.25
    dropout_fc: float = 0.5
    use_batch_norm: bool = True

    def __post_init__(self):
        if self.conv_filters is None:
            self.conv_filters = [32, 64, 128]


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = "crossentropyloss"
    lr_scheduler: str = "none"
    lr_decay_step: int = 5
    lr_decay_factor: float = 0.1
    early_stopping: bool = False
    patience: int = 10
    save_frequency: int = 5
    checkpoint_dir: str = "./checkpoints"


@dataclass
class DeviceConfig:
    """Device configuration."""
    type: str = "auto"
    mixed_precision: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    output_dir: str = "./outputs"
    save_plots: bool = True
    plot_types: list = None

    def __post_init__(self):
        if self.plot_types is None:
            self.plot_types = ["training_history", "confusion_matrix", "sample_predictions"]


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str = "urban_scene_cnn_v1"
    description: str = "CNN for urban scene classification"
    tags: list = None
    use_wandb: bool = False

    def __post_init__(self):
        if self.tags is None:
            self.tags = ["cnn", "image-classification", "pytorch"]


class Config:
    """Main configuration class."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from YAML file or defaults.
        
        Args:
            config_path: Path to config YAML file. If None, uses defaults.
        """
        self.dataset = DatasetConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.device = DeviceConfig()
        self.logging = LoggingConfig()
        self.experiment = ExperimentConfig()
        self.random_seed = 42

        if config_path:
            self.load_from_yaml(config_path)

        self._setup_output_dirs()

    def load_from_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file.
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

        # Update configurations from YAML
        if "dataset" in config_dict:
            self.dataset = DatasetConfig(**config_dict["dataset"])
        if "model" in config_dict:
            self.model = ModelConfig(**config_dict["model"])
        if "training" in config_dict:
            self.training = TrainingConfig(**config_dict["training"])
        if "device" in config_dict:
            self.device = DeviceConfig(**config_dict["device"])
        if "logging" in config_dict:
            self.logging = LoggingConfig(**config_dict["logging"])
        if "experiment" in config_dict:
            self.experiment = ExperimentConfig(**config_dict["experiment"])
        if "random_seed" in config_dict:
            self.random_seed = config_dict["random_seed"]

        logger.info(f"Configuration loaded from {config_path}")

    def _setup_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.logging.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.dataset.path).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration.
        """
        return {
            "dataset": asdict(self.dataset),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "device": asdict(self.device),
            "logging": asdict(self.logging),
            "experiment": asdict(self.experiment),
            "random_seed": self.random_seed,
        }

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(\n{yaml.dump(self.to_dict(), default_flow_style=False)})"


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration instance.
    
    Args:
        config_path: Path to config file. If None, uses default config.
        
    Returns:
        Config instance.
    """
    if config_path is None:
        config_path = "configs/default.yaml"

    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return Config()

    return Config(config_path)
