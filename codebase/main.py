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

    # Get dataset sizes
    total = len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)

    # Print comprehensive results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    # Dataset split table
    print("\nDataset Split:")
    print("-" * 40)
    print(f"{'Split':<12} {'Images':<10} {'Benign':<10} {'Malignant':<10}")
    print("-" * 40)
    print(f"{'Total':<12} {total:<10} {total//2}(50%)    {total//2}(50%)")
    print(f"{'Train':<12} {train_size:<10} {train_size//2:<10} {train_size//2:<10}")
    print(f"{'Validation':<12} {val_size:<10} {val_size//2:<10} {val_size//2:<10}")
    print(f"{'Test':<12} {test_size:<10} {test_size//2:<10} {test_size//2:<10}")
    print("\nData Source: HAM10000 (Human Against Machine with 10,000 training images)")
    print(f"             Subset of {total} images")
    
    # Quantitative results table
    print("\n" + "="*60)
    print("QUANTITATIVE RESULTS")
    print("="*60)
    print(f"\n{'Metric':<20} {'Value':<10}")
    print("-" * 30)
    print(f"{'Test Accuracy':<20} {metrics['accuracy']*100:.1f}%")
    print(f"{'Train Accuracy':<20} {history['train_acc'][-1]*100:.1f}%")
    print(f"{'Validation Accuracy':<20} {history['val_acc'][-1]*100:.1f}%")
    print(f"{'Precision':<20} {metrics['precision']*100:.1f}%")
    print(f"{'Recall':<20} {metrics['recall']*100:.1f}%")
    print(f"{'F1-Score':<20} {metrics['f1_score']*100:.1f}%")
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    print(f"\n{'':20} {'Pred Benign':<15} {'Pred Malignant':<15}")
    print("-" * 50)
    print(f"{'REAL Benign':<20} {cm[0][0]:<15} {cm[0][1]:<15}")
    print(f"{'REAL Malignant':<20} {cm[1][0]:<15} {cm[1][1]:<15}")
    print(f"\nFalse Negatives: {cm[1][0]} | False Positives: {cm[0][1]}")
    
    print("\n" + "="*60)
    print(f"Best model saved to: {config.checkpoint_path}")
    print("="*60)


if __name__ == "__main__":
    main()
