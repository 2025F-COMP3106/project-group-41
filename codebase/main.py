"""
Entry point for training the skin cancer prediction agent.
Run from project root: python -m codebase.main
Or directly: python codebase/main.py
"""

import sys
import os

# Add project root to path so imports work when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codebase.config import TrainingConfig, ModelConfig
from codebase.training import Trainer
from codebase.models import ResNetModel
from codebase.data.dataset_loader import get_dataloaders


def main():
    print("\n" + "="*60)
    print("SKIN CANCER DETECTION - NEURAL NETWORK TRAINING")
    print("="*60)

    # Training configuration (uses defaults from config.py)
    config = TrainingConfig()
    print(f"\nDevice: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")

    # Load real data
    print("\nLoading HAM10000 dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Create model - ResNet18 with pretrained ImageNet weights (uses defaults from config.py)
    model_config = ModelConfig()
    
    model = ResNetModel(model_config.to_dict())
    print(f"\nModel: {model.__class__.__name__} (pretrained=True)")

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # Train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    history = trainer.fit()

    # Evaluate on test set
    print("\n" + "="*60)
    print("EVALUATION (Test Set)")
    print("="*60)
    metrics = trainer.evaluate(test_loader)

    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Train Accuracy: {history['train_acc'][-1]*100:.1f}%")
    print(f"Val Accuracy:   {history['val_acc'][-1]*100:.1f}%")
    print(f"Test Accuracy:  {metrics['accuracy']*100:.1f}%")
    print(f"\nBest model saved to: {config.checkpoint_path}")


if __name__ == "__main__":
    main()
