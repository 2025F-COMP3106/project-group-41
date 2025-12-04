# Helper functions for data processing, transforms, and utilities

import numpy as np
import torch
import random
import os
from torchvision import transforms


# =============================================================================
# BATCH UTILITIES
# =============================================================================

def create_batches(X, y, batch_size):
    """
    Create mini-batches from data for training.

    Parameters:
        X          : features 
        y          : labels 
        batch_size : Size of each batch
        
    Yields:
        batch_X, batch_y : (batch_X, batch_y) for each batch
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        batch_X = X[batch_indices]
        batch_y = y[batch_indices]
        yield batch_X, batch_y


# =============================================================================
# IMAGE TRANSFORMS
# =============================================================================

def get_preprocessing_transforms():
    """
    Returns standard preprocessing transforms for medical images.
    Resizes to 224x224, converts to tensor, normalizes using ImageNet stats.
    
    Returns:
        torchvision.transforms.Compose: Preprocessing transforms
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_train_transforms():
    """
    Returns augmentation transforms for training data.
    Includes random flips, rotations, color jitter, and affine transformations
    to increase data diversity and prevent overfitting.
    
    Returns:
        torchvision.transforms.Compose: Composition of augmentation transforms
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_test_transforms():
    """
    Returns transforms for validation and test data.
    No augmentation, only resizing, tensor conversion, and normalization.
    
    Returns:
        torchvision.transforms.Compose: Composition of preprocessing transforms
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =============================================================================
# SEED UTILITIES
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get available device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')