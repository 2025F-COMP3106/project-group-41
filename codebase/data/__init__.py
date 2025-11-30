# Data module - Export main functions for easy importing

# Import main functions/classes from each data module
from .dataset_loader import get_dataloaders, SkinLesionDataset, split_dataset
# - get_dataloaders: Main function to create train/val/test DataLoaders
# - SkinLesionDataset: Custom PyTorch Dataset class for loading images
# - split_dataset: Function to split data into train/val/test with stratification

from .augmentation import get_train_transforms, get_val_test_transforms
# - get_train_transforms: Augmentation transforms for training (flips, rotations, etc.)
# - get_val_test_transforms: Preprocessing transforms for validation/test (no augmentation)

from .preprocessor import get_preprocessing_transforms
# - get_preprocessing_transforms: Basic preprocessing (resize, normalize, tensor conversion)

# __all__ explicitly defines what gets exported when someone does: from codebase.data import *
# This just basically  prevents importing internal/private functions and makes the API clear
__all__ = [
    'get_dataloaders',           # Most commonly used - creates DataLoaders
    'SkinLesionDataset',         # Custom dataset class
    'split_dataset',             # Data splitting utility
    'get_train_transforms',      # Training augmentation
    'get_val_test_transforms',   # Validation/test preprocessing
    'get_preprocessing_transforms'  # Basic preprocessing
]
