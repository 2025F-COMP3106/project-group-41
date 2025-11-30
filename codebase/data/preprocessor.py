# Preprocessing pipeline for medical images
# 
# Purpose: Prepares raw images for the neural network by standardizing format
# (resize to 224x224, convert to tensor, normalize). This ensures all images
# are in the same format the model expects, regardless of original size/format.

import torch
from torchvision import transforms

# torchvision.transforms: Image transformation functions for PyTorch
# Provides: resize, normalize, tensor conversion

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