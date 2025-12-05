# Training loop and logic

import torch
import numpy as np

from torch.nn import Module
from torch.utils.data import DataLoader

from codebase.training.callbacks import EarlyStopping, ModelCheckpoint
from codebase.config import TrainingConfig
from codebase.evaluation import calculate_metrics


class Trainer:
    """
    Trainer class: connects data, model, and optimization.
    Handles:
      - training loop
      - validation after each epoch
      - early stopping
      - checkpoint saving
      - metric logging
    """

    def __init__(
        self,
        model: Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Device (GPU/CPU)
        self.device = config.device
        self.model.to(self.device)

        # Loss function & optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
        )
        self.checkpoint = ModelCheckpoint(config.checkpoint_path)

        # History dictionary (for plotting learning curves later)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    # Core training method

    def fit(self):
        """
        Full training loop across all epochs.
        After each epoch:
          - compute train loss & accuracy
          - compute validation loss & accuracy
          - update callbacks
        """
        print(f"[Trainer] Starting training for {self.config.num_epochs} epochs...")
        for epoch in range(self.config.num_epochs):

            # 1. Train for one epoch
            train_loss, train_metrics = self._run_one_epoch(self.train_loader, train=True)

            # 2. Validate
            val_loss, val_metrics = self._run_one_epoch(self.val_loader, train=False)

            # 3. Log metrics in history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            # 4. Print progress
            print(
                f"Epoch [{epoch+1}/{self.config.num_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f} "
                f"|| Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # 5. Callbacks (checkpoint + early stopping)
            self.checkpoint(self.model, val_loss)
            self.early_stopping(val_loss)

            if self.early_stopping.should_stop:
                print(
                    f"[Trainer] Early stopping triggered after epoch {epoch+1}. "
                    f"Best validation loss: {self.early_stopping.best_loss:.4f}"
                )
                break

        print("[Trainer] Training finished.")
        return self.history

    # ---------- Helper to run one epoch (train or val) ----------

    def _run_one_epoch(self, loader: DataLoader, train: bool = True):
        """
        Run a single epoch on either the training or validation set.

        Returns:
            avg_loss    : Mean loss over all batches
            metrics     : Dict with accuracy, precision, recall, etc.
        """
        if train:
            self.model.train()
        else:
            self.model.eval()

        running_loss = 0.0
        n_samples = 0

        all_preds = []
        all_labels = []

        for batch in loader:
            # Expecting loader to yield (images, labels)
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            batch_size = inputs.size(0)

            # Zero gradients only for training
            if train:
                self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)               # shape: (batch_size, num_classes)
            loss = self.criterion(outputs, labels)

            # Backward + update only in training mode
            if train:
                loss.backward()
                self.optimizer.step()

            # Track stats
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            # Store predictions & labels (on CPU for sklearn metrics)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

        # Concatenate all batches
        y_true = np.concatenate(all_labels, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)

        avg_loss = running_loss / max(n_samples, 1)

        # Use your existing metrics helper
        metrics = calculate_metrics(y_true, y_pred)

        return avg_loss, metrics

    # ---------- Evaluation on a separate test set ----------

    def evaluate(self, test_loader: DataLoader):
        """
        Evaluate the (trained) model on a held-out test set.
        Returns a metrics dictionary (accuracy, recall, etc.).
        """
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        y_true = np.concatenate(all_labels, axis=0)
        y_pred = np.concatenate(all_preds, axis=0)

        metrics = calculate_metrics(y_true, y_pred)
        print("[Trainer] Test set metrics:", metrics)
        return metrics
