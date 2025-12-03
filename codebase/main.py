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
from torch.utils.data import DataLoader, TensorDataset
from codebase.config import TrainingConfig
from codebase.training import Trainer
from codebase.models import SimpleCNN, ResNetModel


def main():
    print("\n" + "="*60)
    print("SKIN CANCER DETECTION - NEURAL NETWORK TRAINING")
    print("="*60)

    # 1. Training configuration
    config = TrainingConfig(
        num_epochs=5,
        learning_rate=1e-3,
        batch_size=8,
        checkpoint_path="checkpoints/best_model.pth",
        early_stopping_patience=5
    )
    print(f"\nDevice: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")

    # 2. Generate synthetic data (since we don't have enough real images yet)
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

    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # 3. Select model
    model_config = {
        "num_classes": 2,
        "input_shape": (3, 224, 224),
        "task": "classification",
        "model_type": "pytorch",
    }
    
    # OPTION A: SimpleCNN (faster, good for testing)
    model = SimpleCNN(model_config)
    
    # OPTION B: ResNet (pretrained, better accuracy)
    # model_config["pretrained"] = True
    # model_config["resnet_version"] = "resnet18"
    # model = ResNetModel(model_config)

    print(f"\nModel: {model.__class__.__name__}")

    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # 5. Train model
    print("\n" + "="*60)
    print("TRAINING")
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
    print(f"\nBest model saved to: {config.checkpoint_path}")


if __name__ == "__main__":
    main()
