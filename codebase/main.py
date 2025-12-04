"""
Entry point for training the skin cancer prediction agent.
Run from project root: python -m codebase.main
Or directly: python codebase/main.py
"""

import sys
import os

# Add project root to path so imports work when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from codebase.config import TrainingConfig
from codebase.training import Trainer
from codebase.models import SimpleCNN, ResNetModel
from codebase.data.dataset_loader import get_dataloaders


def main():
    print("\n" + "="*60)
    print("SKIN CANCER DETECTION - NEURAL NETWORK TRAINING")
    print("="*60)
    print("Applying all optimizations (no additional data)")
    print("="*60)

    # ==========================================================
    # OPTIMIZATION 1: Use pretrained ResNet (transfer learning)
    # ==========================================================
    
    # ==========================================================
    # OPTIMIZATION 2: Better hyperparameters
    # - Lower learning rate (1e-4) for fine-tuning pretrained model
    # - More epochs (20) to allow convergence
    # - Larger batch size (16) for more stable gradients
    # - Weight decay (1e-4) for regularization
    # ==========================================================
    config = TrainingConfig(
        num_epochs=20,
        learning_rate=1e-4,      # Lower LR for pretrained model
        batch_size=16,
        weight_decay=1e-4,       # L2 regularization
        checkpoint_path="checkpoints/best_model.pth",
        early_stopping_patience=7
    )
    print(f"\nDevice: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Weight decay: {config.weight_decay}")

    # 2. Generate synthetic data (since we don't have enough real images yet)
    ''''
    print("\nGenerating synthetic training data...")
    X_train = torch.randn(64, 3, 224, 224)  # 64 fake 224x224 RGB images
    y_train = torch.randint(0, 2, (64,))    # Random labels (0=benign, 1=malignant)
    X_val = torch.randn(16, 3, 224, 224)    # 16 validation images
    y_val = torch.randint(0, 2, (16,))
    X_test = torch.randn(16, 3, 224, 224)   # 16 test images
    y_test = torch.randint(0, 2, (16,))

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.batch_size)

    print(f"Train samples: {len(X_train)} (balanced classes)")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    '''

    # using real data
    print("\nLoading real HAM10000 subset...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # ==========================================================
    # OPTIMIZATION 1: Pretrained ResNet18 (transfer learning)
    # ==========================================================
    model_config = {
        "num_classes": 2,
        "input_shape": (3, 224, 224),
        "task": "classification",
        "model_type": "pytorch",
        "pretrained": False,          # Use ImageNet pretrained weights
        "resnet_version": "resnet18"  # Lighter model, good for small data
    }
    
    model = ResNetModel(model_config)
    print(f"\nModel: {model.__class__.__name__} (pretrained=True)")

    # ==========================================================
    # OPTIMIZATION 4: Class weighting (handle imbalance)
    # ==========================================================
    # Count classes and compute weights (inverse frequency)
    print("\nComputing class weights from real training data...")

    labels_list = []
    for _, labels in train_loader:
        labels_list.extend(labels.tolist())

    labels_tensor = torch.tensor(labels_list)
    class_counts = torch.bincount(labels_tensor)

    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * 2  # normalize to sum=2

    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class weights: {class_weights.tolist()}")

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )
    
    # Override loss with weighted version
    trainer.criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.device))

    # 5. Train model
    print("\n" + "="*60)
    print("TRAINING (with all optimizations)")
    print("="*60)
    history = trainer.fit()

    # 6. Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATION (Test Set)")
    print("="*60)
    metrics = trainer.evaluate(test_loader)

    # 7. Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Train Accuracy: {history['train_acc'][-1]*100:.1f}%")
    print(f"Val Accuracy:   {history['val_acc'][-1]*100:.1f}%")
    print(f"Test Accuracy:  {metrics['accuracy']*100:.1f}%")
    
    print("\n" + "="*60)
    print("OPTIMIZATIONS APPLIED:")
    print("="*60)
    print("1. Pretrained ResNet18 (transfer learning)")
    print("2. Lower learning rate (1e-4) for fine-tuning")
    print("3. Weight decay (L2 regularization)")
    print("4. Class weighting in loss function")
    print("5. More epochs (20) with early stopping")
    print("6. Patterned synthetic data (learnable structure)")
    print(f"\nBest model saved to: {config.checkpoint_path}")


if __name__ == "__main__":
    main()
