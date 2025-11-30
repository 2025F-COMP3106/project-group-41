# Preprocessing pipeline for medical images

import torch
# torchvision.transforms: A collection of image transformation functions for PyTorch
# Why we import it:
#   - Provides ready-made image transformations (resize, normalize, tensor conversion, etc.)
#   - Handles conversion between PIL Images and PyTorch tensors automatically
#   - Allows us to chain multiple transforms together using Compose()
#   - Essential for standardizing images before feeding them to neural networks
#   - Without this, we'd have to manually write all image preprocessing code
from torchvision import transforms


def get_preprocessing_transforms():
    """
    Returns standard preprocessing transforms for medical images.
    Resizes images to 224x224 (standard for ResNet), converts to tensor,
    and normalizes using ImageNet statistics.
    
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

