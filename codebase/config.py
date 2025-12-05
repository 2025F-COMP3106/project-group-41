# training/config.py

import torch

class TrainingConfig:
    #Container for all training-related hyperparameters. This makes experiments easy to reproduce and modify.

    def __init__(
        self,
        num_epochs: int = 20,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        checkpoint_path: str = "checkpoints/best_model.pth",
        early_stopping_patience: int = 5,
        early_stopping_min_delta: float = 1e-4,
        device: str | None = None,
    ):
        # training loop hyperparameters
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        # model saving / early stopping
        self.checkpoint_path = checkpoint_path
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta

        # device selection (GPU if available, else CPU)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device


class ModelConfig:
    #Container for model architecture parameters.

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        resnet_version: str = "resnet18",
    ):
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.resnet_version = resnet_version

    def to_dict(self):
        """Convert config to dictionary for model initialization."""
        return {
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "resnet_version": self.resnet_version,
        }
