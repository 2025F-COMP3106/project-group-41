# Cross-validation strategies (k-fold, stratified k-fold)
# Purpose: Test model performance by training/testing on different data splits
# This helps ensure the model generalizes well and isn't just memorizing data

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from typing import Dict, List, Tuple, Any
import copy

from codebase.evaluation.metrics import calculate_metrics


def k_fold_cross_validation(
    model, 
    X: np.ndarray, 
    y: np.ndarray, 
    k: int = 5, 
    shuffle: bool = True, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.
    Splits data into k parts, trains on k-1, tests on 1, repeats k times.
    
    Args:
        model: Model instance (must have fit() and predict() methods)
        X: Features (n_samples, ...)
        y: Labels (n_samples,)
        k: Number of folds (default 5 = 5 different train/test splits)
        shuffle: Whether to shuffle data before splitting
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with fold scores and averages
    """
    # Create k-fold splitter (divides data into k equal parts)
    kfold = KFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    
    # Store results from each fold
    fold_scores = []
    
    # Loop through each fold (each time, different part is used for testing)
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        # Split data using indices from this fold
        X_train, y_train = X[train_idx], y[train_idx]  # Training data (k-1 parts)
        X_val, y_val = X[val_idx], y[val_idx]          # Validation data (1 part)
        
        # Create fresh copy of model (don't want to reuse trained weights)
        fold_model = copy.deepcopy(model)
        
        # Train model on training data
        fold_model.fit(X_train, y_train)
        
        # Make predictions on validation data
        y_pred = fold_model.predict(X_val)
        
        # Calculate how well the model did on this fold
        metrics = calculate_metrics(y_val, y_pred)
        metrics['fold'] = fold_idx + 1  # Track which fold this is
        fold_scores.append(metrics)
    
    # Combine results from all folds (mean, std)
    return _aggregate_cv_results(fold_scores)


def stratified_k_fold_cross_validation(
    model, 
    X: np.ndarray, 
    y: np.ndarray, 
    k: int = 5, 
    shuffle: bool = True, 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform stratified k-fold cross-validation.
    Same as k-fold, but ensures each fold has same class distribution.
    Better for imbalanced datasets (e.g., 90% benign, 10% malignant).
    
    Args:
        model: Model instance
        X: Features
        y: Labels
        k: Number of folds
        shuffle: Whether to shuffle data
        random_state: Random seed
    
    Returns:
        Dictionary with fold scores and averages
    """
    # Stratified splitter maintains class proportions in each fold
    # If dataset is 70% benign, each fold will also be ~70% benign
    skfold = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
    
    fold_scores = []
    
    # Loop through each fold
    for fold_idx, (train_idx, val_idx) in enumerate(skfold.split(X, y)):
        # Split data (stratified maintains class proportions)
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Fresh model for this fold
        fold_model = copy.deepcopy(model)
        
        # Train and predict
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_val)
        
        # Calculate metrics for this fold
        metrics = calculate_metrics(y_val, y_pred)
        metrics['fold'] = fold_idx + 1
        fold_scores.append(metrics)
    
    return _aggregate_cv_results(fold_scores)


def leave_one_out_validation(model, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Perform leave-one-out cross-validation (k = n_samples).
    For each sample: train on all others, test on that one sample.
    Useful for very small datasets, but slow for large ones.
    
    Args:
        model: Model instance
        X: Features
        y: Labels
    
    Returns:
        Validation results
    """
    n_samples = len(X)
    predictions = np.zeros(n_samples)  # Store prediction for each sample
    
    # For each sample, train on all others and predict this one
    for i in range(n_samples):
        # Create mask: True for training samples, False for test sample
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[i] = False  # Exclude sample i from training
        
        # Split data
        X_train, y_train = X[train_mask], y[train_mask]  # All except sample i
        X_test = X[i:i+1]  # Just sample i (single sample)
        
        # Fresh model, train, predict
        fold_model = copy.deepcopy(model)
        fold_model.fit(X_train, y_train)
        predictions[i] = fold_model.predict(X_test)[0]
    
    # Calculate overall metrics (comparing all predictions to true labels)
    metrics = calculate_metrics(y, predictions.astype(int))
    metrics['n_samples'] = n_samples
    
    return metrics


def train_val_test_split(
    X: np.ndarray, 
    y: np.ndarray, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15, 
    stratify: bool = True, 
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    Standard split for model development:
    - Train: model learns from this
    - Validation: tune hyperparameters, check for overfitting
    - Test: final evaluation (only use once at the end)
    
    Args:
        X: Features
        y: Labels
        train_ratio: Proportion for training (default: 0.7 = 70%)
        val_ratio: Proportion for validation (default: 0.15 = 15%)
        test_ratio: Proportion for testing (default: 0.15 = 15%)
        stratify: Whether to maintain class distribution in each split
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # Make sure ratios add up to 100%
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    # If stratify=True, maintain class proportions in each split
    stratify_labels = y if stratify else None
    
    # First split: separate training data from (validation + test)
    # Result: 70% train, 30% temp (which will become val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=(val_ratio + test_ratio),  # 0.15 + 0.15 = 0.30
        stratify=stratify_labels, 
        random_state=random_state
    )
    
    # Second split: separate validation from test
    # From the 30% temp, split into 15% val and 15% test
    val_size = val_ratio / (val_ratio + test_ratio)  # 0.15 / 0.30 = 0.5
    stratify_temp = y_temp if stratify else None
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, 
        test_size=(1 - val_size),  # 0.5 goes to test
        stratify=stratify_temp, 
        random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def _aggregate_cv_results(fold_scores: List[Dict]) -> Dict[str, Any]:
    """
    Combine results from all cross-validation folds.
    Calculates mean and standard deviation across folds.
    
    Args:
        fold_scores: List of metric dictionaries from each fold
    
    Returns:
        Dictionary with fold_scores, mean, and std for each metric
    """
    # Metrics we want to aggregate
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    results = {'fold_scores': fold_scores}  # Keep individual fold results
    
    # For each metric, calculate mean and std across folds
    for metric in metric_names:
        values = [fold[metric] for fold in fold_scores if metric in fold]
        if values:
            results[f'mean_{metric}'] = np.mean(values)  # Average performance
            results[f'std_{metric}'] = np.std(values)    # How much it varies
    
    return results
