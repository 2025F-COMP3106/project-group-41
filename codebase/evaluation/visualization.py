# Visualization of results (learning curves, confusion matrices, ROC curves)
# Purpose: Create plots to understand model performance visually
# These help identify issues like overfitting and show where the model makes mistakes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from typing import Dict, List, Optional, Any


def plot_learning_curves(history: Dict[str, List], save_path: Optional[str] = None):
    """
    Plot training and validation loss/accuracy curves over epochs.
    Shows how the model improved during training and if it's overfitting.
    
    Overfitting signs: train loss decreases but val loss increases
    Good training: both train and val loss decrease together
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
        save_path: Path to save figure (if None, display instead)
    
    Returns:
        Matplotlib figure object
    """
    # Create figure with 2 side-by-side plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # X-axis: epoch numbers (1, 2, 3, ...)
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Left plot: Loss over time
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Training')   # Blue = training
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')   # Red = validation
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)  # Light grid for readability
    
    # Right plot: Accuracy over time
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Training')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()  # Prevent overlapping
    
    # Save or display
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: Optional[List[str]] = None, 
    normalize: bool = False, 
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix as a heatmap.
    Shows where the model makes correct predictions and mistakes.
    
    Matrix layout:
                    Predicted
                    Benign  Malignant
    True  Benign    [TN]    [FP]      <- False Positives (predicted cancer but wasn't)
          Malignant [FN]    [TP]      <- False Negatives (missed cancer - dangerous!)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names (default: ['Benign', 'Malignant'])
        normalize: Whether to show percentages instead of counts
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Default class names for skin cancer
    if class_names is None:
        class_names = ['Benign', 'Malignant']
    
    # Compute confusion matrix (rows=true, cols=predicted)
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize to percentages if requested
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Divide by row totals
        fmt = '.2%'  # Show as percentage
    else:
        fmt = 'd'  # Show as integer
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create heatmap (darker blue = higher value)
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)  # Add color scale
    
    # Set axis labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix')
    
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add numbers in each cell
    thresh = cm.max() / 2.  # Threshold for text color (white on dark, black on light)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = f'{cm[i, j]:{fmt}}' if normalize else f'{cm[i, j]}'
            ax.text(j, i, value, ha='center', va='center',
                   color='white' if cm[i, j] > thresh else 'black', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_roc_curve(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    save_path: Optional[str] = None
):
    """
    Plot ROC (Receiver Operating Characteristic) curve for binary classification.
    Shows trade-off between True Positive Rate and False Positive Rate.
    
    - AUC = 1.0: Perfect classifier
    - AUC = 0.5: Random guessing (diagonal line)
    - AUC > 0.8: Generally considered good
    
    Args:
        y_true: True labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Handle 2D probability array (take probability of positive class)
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[:, 1]  # Column 1 = malignant probability
    
    # Calculate ROC curve points
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)  # FPR and TPR at different thresholds
    roc_auc = auc(fpr, tpr)  # Area Under Curve (higher = better)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    
    # Plot diagonal (random classifier baseline)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')  # How often we wrongly predict positive
    ax.set_ylabel('True Positive Rate')   # How often we correctly predict positive (Recall)
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray, 
    save_path: Optional[str] = None
):
    """
    Plot Precision-Recall curve.
    Better than ROC for imbalanced datasets (e.g., rare cancers).
    
    - Precision: Of all predicted positives, how many are correct?
    - Recall: Of all actual positives, how many did we find?
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities for positive class
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Handle 2D probability array
    if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] == 2:
        y_pred_proba = y_pred_proba[:, 1]
    
    # Calculate precision and recall at different thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Find best threshold (highest F1 score = balance of precision and recall)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
    best_idx = np.argmax(f1_scores)  # Index of best F1
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot PR curve
    ax.plot(recall, precision, 'b-', lw=2, label='Precision-Recall Curve')
    
    # Mark the best F1 point
    ax.scatter(recall[best_idx], precision[best_idx], c='red', s=100, 
               label=f'Best F1 = {f1_scores[best_idx]:.3f}', zorder=5)
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')     # How many actual positives we found
    ax.set_ylabel('Precision')  # How accurate our positive predictions are
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_feature_importance(
    importances: np.ndarray, 
    feature_names: Optional[List[str]] = None, 
    top_k: int = 20, 
    save_path: Optional[str] = None
):
    """
    Plot feature importance as horizontal bar chart.
    Shows which features the model relies on most for predictions.
    Useful for interpretable models (Random Forest, Logistic Regression).
    
    Args:
        importances: Feature importance values (from model)
        feature_names: List of feature names
        top_k: Number of top features to display
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Default feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    # Sort features by importance (highest first)
    indices = np.argsort(importances)[::-1][:top_k]  # Get top_k indices
    top_importances = importances[indices]
    top_names = [feature_names[i] for i in indices]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.3)))
    
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()  # Most important at top
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_k} Feature Importances')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig


def plot_model_comparison(
    models_results: Dict[str, float], 
    metric: str = 'accuracy', 
    save_path: Optional[str] = None
):
    """
    Compare multiple models on a single metric.
    Useful for deciding which model to use.
    
    Args:
        models_results: Dictionary mapping model_name -> metric_value
                       e.g., {'SimpleCNN': 0.85, 'ResNet': 0.92}
        metric: Metric name to display (e.g., 'accuracy', 'f1_score')
        save_path: Path to save figure
    
    Returns:
        Matplotlib figure
    """
    # Sort models by score (best first)
    sorted_items = sorted(models_results.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    # Color gradient: green (best) to red (worst)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(names)))[::-1]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create bar chart
    bars = ax.bar(names, scores, color=colors)
    
    # Add score labels on top of each bar
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add baseline (random guessing = 0.5 for binary classification)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random Baseline')
    
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'Model Comparison - {metric.capitalize()}')
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')  # Rotate model names
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    return fig
