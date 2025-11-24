"""
Entry point for training the skin cancer prediction agent.
"""

import torch
from codebase.config import TrainingConfig
from codebase.training.trainer import Trainer


from codebase.models.neural_networks import SimpleCNN, ResNetModel
from codebase.data.dataset_loader import get_dataloaders


def main():
    print("\n========== Skin Cancer Detection Training ==========\n")


    # 1. Training configuration
    config = TrainingConfig(
        num_epochs=20,
        learning_rate=1e-4,
        batch_size=32,
        checkpoint_path="checkpoints/best_model.pth",
        early_stopping_patience=5
    )
    print(f"Using device: {config.device}")

    # 2. Load dataset
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(config)
    print("Train/Val/Test loaders ready.\n")

    # 3. Select model
    # OPTION A: Simple CNN
    # model = SimpleCNN(num_classes=2)

    # OPTION B: ResNet
    model_config = {
        "num_classes": 2,
        "pretrained": True,
        "resnet_version": "resnet18",
        "task": "classification",
        "model_type": "pytorch",
    }
    model = ResNetModel(model_config)

    print(f"Model initialized: {model.__class__.__name__}\n")

    # 4. Initialize Trainer (Izu's part)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    # 5. Train model

    print("[Training] Started...\n")
    history = trainer.fit()

    # 6. Evaluate final best model

    print("\n[Evaluation] Running on test set...\n")

    metrics = trainer.evaluate(test_loader)
    print("\nFinal Test Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    print("\nTraining complete. Best model saved to:")
    print(f"  {config.checkpoint_path}\n")


if __name__ == "__main__":
    main()
