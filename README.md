# Urban Scene CNN

A convolutional neural network implementation for classifying urban scenes.

## Overview

This is a CNN built with PyTorch to classify different urban scene categories. The code is organized into separate modules for data handling, model architecture, training, and utilities. Configuration is managed through YAML files.

The model uses batch normalization and dropout for regularization. Architecture consists of 3 convolutional blocks followed by fully connected layers. Total parameters: ~16.8M.

## Setup

Clone the repository:
```bash
git clone https://github.com/Hassan-Naeem-code/Computer-Vision-Week-03.git
cd Computer-Vision-Week-03
```

Install dependencies:
```bash
pip install -e .
```

Or with dev dependencies:
```bash
pip install -e ".[dev]"
```

## Running

To train the model:
```bash
python -m src.main --config configs/default.yaml
```

The script will automatically create a dummy dataset if the real dataset is not found.

## Configuration

Edit `configs/default.yaml` to adjust:
- Training parameters (epochs, batch size, learning rate)
- Model architecture (filter sizes, dropout rates)
- Data split ratios
- Output directories

You can also use a custom config file:
```bash
python -m src.main --config configs/your_config.yaml
```

## Testing

Run tests:
```bash
pytest tests/ -v
```

All 14 tests cover the dataset loading, model architecture, and training utilities.

## Model

The CNN consists of:
- 3 convolutional blocks (32, 64, 128 filters)
- Batch normalization after each layer
- Dropout for regularization
- 2 fully connected layers at the end

Each convolutional block includes: Conv2D → BatchNorm → ReLU → MaxPool → Dropout

## Results

After training, the model generates:
- `training_history.png` - Training and validation loss/accuracy
- `test_accuracy.png` - Test set performance
- `confusion_matrix.png` - Per-class breakdown
- `checkpoints/best_model.pth` - Saved model weights

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib, scikit-learn
- See requirements.txt for full list

## License

MIT
