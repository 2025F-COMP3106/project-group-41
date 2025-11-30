# Data augmentation strategies

import torch

from torchvision import transforms

# torchvision.transforms: A collection of image transformation functions for PyTorch
# Its function:
#   - Provides ready-made image transformations (flip, rotate, resize, normalize, etc.)
#   - Handles conversion between PIL Images and PyTorch tensors
#   - Allows us to chain multiple transforms together using Compose()
#   - Essential for preprocessing images before feeding them to neural networks  - Without this, 
# we'd have to manually write all image manipulation code


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