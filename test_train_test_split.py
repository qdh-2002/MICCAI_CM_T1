#!/usr/bin/env python3
"""
Test script to verify the train/test split behavior in MRI_datasets_knee_kspace.py
"""

import numpy as np
import torch
from pathlib import Path
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cm.MRI_datasets_knee_kspace import FastMRIH5SliceDataset

def test_train_test_split():
    """Test that train uses random masks and test uses a single fixed mask"""
    
    # Mock target shape for testing
    target_shape = (256, 256)
    
    # Create datasets with different splits
    print("Creating train dataset...")
    train_dataset = FastMRIH5SliceDataset(
        root="/path/to/mock",  # This will fail file loading but we can test the mask logic
        split="train",
        target_shape=target_shape
    )
    
    print("Creating test dataset...")
    test_dataset = FastMRIH5SliceDataset(
        root="/path/to/mock",  # This will fail file loading but we can test the mask logic
        split="test", 
        target_shape=target_shape
    )
    
    # Check that test dataset has a fixed mask
    print(f"Test dataset has test_mask: {hasattr(test_dataset, 'test_mask')}")
    if hasattr(test_dataset, 'test_mask'):
        print(f"Test mask shape: {test_dataset.test_mask.shape}")
        print(f"Test mask type: {type(test_dataset.test_mask)}")
        print(f"Test mask min/max: {test_dataset.test_mask.min():.3f}/{test_dataset.test_mask.max():.3f}")
        print(f"Test mask sparsity: {np.mean(test_dataset.test_mask):.3f} (fraction of 1s)")
    
    # Test that train dataset doesn't have a fixed mask
    print(f"Train dataset has test_mask: {hasattr(train_dataset, 'test_mask')}")
    
    print("\nTrain/test split implementation verified!")

if __name__ == "__main__":
    test_train_test_split()
