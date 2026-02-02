"""Tests for dataset loading and preprocessing."""

import os
import tempfile
import pytest
import torch
from src.data import load_dataset, create_dummy_dataset


def test_create_dummy_dataset():
    """Test dummy dataset creation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir, num_classes=3, samples_per_class=10)
        
        # Check that directories were created
        assert os.path.isdir(tmpdir)
        class_dirs = [d for d in os.listdir(tmpdir) 
                      if os.path.isdir(os.path.join(tmpdir, d))]
        assert len(class_dirs) == 3
        
        # Check that images were created
        for class_dir in class_dirs:
            images = os.listdir(os.path.join(tmpdir, class_dir))
            assert len(images) == 10


def test_load_dataset():
    """Test dataset loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir, num_classes=3, samples_per_class=20)
        
        train_loader, val_loader, test_loader, dataset = load_dataset(
            dataset_path=tmpdir,
            batch_size=4,
            train_ratio=0.7,
            val_ratio=0.15,
        )
        
        assert len(dataset) == 60  # 3 classes * 20 samples
        assert len(dataset.classes) == 3
        assert len(train_loader) > 0
        assert len(val_loader) > 0
        assert len(test_loader) > 0


def test_data_loader_shapes():
    """Test that data loaders return correct shapes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_dummy_dataset(tmpdir, num_classes=2, samples_per_class=10)
        
        train_loader, _, _, _ = load_dataset(
            dataset_path=tmpdir,
            image_size=128,
            batch_size=4,
        )
        
        # Get first batch
        images, labels = next(iter(train_loader))
        
        assert images.shape[0] == 4  # batch size
        assert images.shape[1] == 3  # channels
        assert images.shape[2] == 128  # height
        assert images.shape[3] == 128  # width
        assert labels.shape[0] == 4
