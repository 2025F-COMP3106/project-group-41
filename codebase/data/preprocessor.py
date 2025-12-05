# Preprocessing transforms for validation and test data

from torchvision import transforms


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
