"""Tests for training utilities."""

import pytest
import torch
from src.training.utils import set_seed, get_device, EarlyStopping


def test_set_seed():
    """Test random seed setting."""
    set_seed(42)
    
    # Generate random numbers
    val1 = torch.randn(5).numpy()
    
    set_seed(42)
    val2 = torch.randn(5).numpy()
    
    # Both should be identical
    assert (val1 == val2).all()


def test_get_device_auto():
    """Test automatic device selection."""
    device = get_device("auto")
    assert isinstance(device, torch.device)


def test_get_device_cpu():
    """Test CPU device selection."""
    device = get_device("cpu")
    assert device.type == "cpu"


def test_early_stopping():
    """Test early stopping functionality."""
    es = EarlyStopping(patience=3)
    
    # Simulate decreasing loss
    assert not es(0.5)
    assert not es(0.4)
    assert not es(0.3)
    assert not es(0.3)  # No improvement
    assert not es(0.3)  # No improvement
    assert es(0.3)      # Should trigger (patience exceeded)


def test_early_stopping_reset():
    """Test early stopping reset."""
    es = EarlyStopping(patience=2)
    
    es(0.5)
    es(0.4)
    assert es(0.4)  # Trigger
    
    es.reset()
    assert es.early_stop is False
    assert es.counter == 0
    assert es.best_value is None
