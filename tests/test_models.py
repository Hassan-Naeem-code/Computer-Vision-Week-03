"""Tests for model architecture."""

import pytest
import torch
from src.models import UrbanSceneCNN


def test_model_initialization():
    """Test model initialization."""
    model = UrbanSceneCNN(num_classes=5)
    assert model is not None
    assert model.num_classes == 5


def test_model_forward_pass():
    """Test model forward pass."""
    model = UrbanSceneCNN(num_classes=5)
    model.eval()
    
    # Create dummy input
    x = torch.randn(2, 3, 128, 128)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (2, 5)  # batch_size=2, num_classes=5


def test_model_parameters():
    """Test model parameter count."""
    model = UrbanSceneCNN(num_classes=5)
    params = model.count_parameters()
    
    assert params["total"] > 0
    assert params["trainable"] > 0
    assert params["total"] >= params["trainable"]


def test_model_summary():
    """Test model summary generation."""
    model = UrbanSceneCNN(num_classes=5)
    summary = model.summary()
    
    assert "UrbanSceneCNN" in summary
    assert "Parameters" in summary


def test_model_training_mode():
    """Test model training and eval modes."""
    model = UrbanSceneCNN(num_classes=5)
    
    # Test training mode
    model.train()
    assert model._is_training is True
    
    # Test eval mode
    model.eval()
    assert model._is_training is False


def test_different_num_classes():
    """Test model with different number of classes."""
    for num_classes in [2, 5, 10, 20]:
        model = UrbanSceneCNN(num_classes=num_classes)
        x = torch.randn(1, 3, 128, 128)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape[1] == num_classes
