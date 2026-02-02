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
    es = EarlyStopping(patience=2)
    
    # Simulate decreasing loss (improvement means lower value)
    # best_value=None, sets to 0.5
    assert not es(0.5)
    # 0.4 < 0.5, counter increases to 1 (no improvement)
    assert not es(0.4)
    # 0.3 < 0.5, counter increases to 2 (no improvement, triggers at patience=2)
    assert es(0.3)


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
