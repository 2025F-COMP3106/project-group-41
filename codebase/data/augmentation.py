# Data augmentation strategies

import torch
from torchvision import transforms

# torchvision.transforms: Image transformation functions for PyTorch
# Provides: flip, rotate, resize, normalize, tensor conversion
# Chains transforms together using Compose()

def get_train_transforms():
    """
    Returns augmentation transforms for training data.
    Includes random flips, rotations, color jitter, and affine transformations
    to increase data diversity and prevent overfitting.
    
    What each transform does:
    - RandomHorizontalFlip: Randomly flips image left/right (50% chance)
    - RandomRotation: Rotates image by random angle (±15 degrees)
    - ColorJitter: Randomly adjusts brightness, contrast, saturation, hue
    - RandomAffine: Applies random translation and scaling
    - Resize: Resizes image to 224x224 (required for ResNet)
    - ToTensor: Converts PIL Image to PyTorch tensor (0-255 -> 0.0-1.0)
    - Normalize: Normalizes tensor using ImageNet statistics
    
    Returns:
        torchvision.transforms.Compose: Composition of augmentation transforms
    """
    return transforms.Compose([
        # Data augmentation transforms (applied randomly during training)
        transforms.RandomHorizontalFlip(p=0.5),  # 50% chance to flip horizontally
        transforms.RandomRotation(degrees=15),     # Rotate by ±15 degrees
        transforms.ColorJitter(                    # Randomly adjust colors
            brightness=0.2,                        # ±20% brightness change
            contrast=0.2,                          # ±20% contrast change
            saturation=0.2,                        # ±20% saturation change
            hue=0.1                                # ±10% hue shift
        ),
        transforms.RandomAffine(                   # Random geometric transformations
            degrees=0,                             # No rotation (already handled above)
            translate=(0.1, 0.1),                  # Shift by up to 10% in x/y
            scale=(0.9, 1.1)                        # Scale between 90%-110%
        ),
        # Preprocessing transforms (always applied)
        transforms.Resize((224, 224)),             # Resize to 224x224 for ResNet
        transforms.ToTensor(),                      # Convert PIL Image to tensor [0.0-1.0]
        transforms.Normalize(                       # Normalize using ImageNet stats
            mean=[0.485, 0.456, 0.406],           # ImageNet mean (RGB channels)
            std=[0.229, 0.224, 0.225]             # ImageNet std (RGB channels)
        )
    ])

def get_val_test_transforms():
    """
    Returns transforms for validation and test data.
    No augmentation, only resizing, tensor conversion, and normalization.
    
    The reason why there is no augmentation here?
    - Validation/test data should be evaluated on original images
    - Augmentation would make results inconsistent and harder to interpret
    - We want to see how well the model performs on real, unmodified images
    
    Returns:
        torchvision.transforms.Compose: Composition of preprocessing transforms
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),             # Resize to 224x224 for ResNet
        transforms.ToTensor(),                      # Convert PIL Image to tensor [0.0-1.0]
        transforms.Normalize(                       # Normalize using ImageNet stats
            mean=[0.485, 0.456, 0.406],           # ImageNet mean (RGB channels)
            std=[0.229, 0.224, 0.225]             # ImageNet std (RGB channels)
        )
    ])