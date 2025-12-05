# Data augmentation transforms for training

from torchvision import transforms


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
