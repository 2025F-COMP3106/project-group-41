# Visualization of results (learning curves, confusion matrices, feature importance)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_learning_curves(history, save_path=None):
    """
    Plot training and validation loss/accuracy curves.
    
    Parameters:
        history   : Dictionary with 'train_loss', 'val_loss', 'train_acc', 'val_acc' lists
        save_path : Path to save figure (if None, display instead)
    
    Returns:
        fig       : Matplotlib figure object
    
    Pseudocode:
        create figure with 2 subplots (1 row, 2 cols)
        
        subplot 1 (Loss):
            plot train_loss vs epochs (blue line)
            plot val_loss vs epochs (orange line)
            add xlabel='Epoch', ylabel='Loss'
            add legend ['Training', 'Validation']
            add title='Model Loss Over Time'
        
        subplot 2 (Accuracy):
            plot train_acc vs epochs (blue line)
            plot val_acc vs epochs (orange line)
            add xlabel='Epoch', ylabel='Accuracy'
            add legend ['Training', 'Validation']
            add title='Model Accuracy Over Time'
        
        if val_loss increases while train_loss decreases:
            mark overfitting region (shaded area or annotation)
        
        if save_path:
            save figure to file
        else:
            display figure
        
        return fig
    """
    pass

def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, save_path=None):
    """
    Plot confusion matrix as a heatmap.
    
    Parameters:
        y_true      : True labels
        y_pred      : Predicted labels
        class_names : List of class names for axes
        normalize   : Whether to show percentages instead of counts
        save_path   : Path to save figure
    
    Returns:
        fig         : Matplotlib figure
    
    Pseudocode:
        compute confusion matrix from y_true and y_pred
        
        if normalize:
            divide each row by row sum (percentage per true class)
        
        create heatmap using seaborn or matplotlib:
            colors: white (0) to dark blue (max)
            annotate each cell with value
            add colorbar
        
        set x-axis label = 'Predicted Label'
        set y-axis label = 'True Label'
        
        if class_names provided:
            set tick labels to class names
        
        add title = 'Confusion Matrix'
        
        if save_path:
            save figure
        
        return fig
    """
    pass

def plot_roc_curve(y_true, y_pred_proba, class_names=None, save_path=None):
    """
    Plot ROC (Receiver Operating Characteristic) curve.
    
    Parameters:
        y_true       : True labels
        y_pred_proba : Predicted probabilities (n_samples, n_classes)
        class_names  : Class names for legend
        save_path    : Path to save figure
    
    Returns:
        fig          : Matplotlib figure
    
    Pseudocode:
        create figure
        
        if binary classification:
            compute FPR (false positive rate) and TPR (true positive rate)
            plot single ROC curve
            compute AUC (area under curve)
            add diagonal reference line (random classifier)
        
        elif multi-class:
            for each class:
                treat as one-vs-rest binary problem
                compute FPR and TPR for this class
                plot ROC curve with class label
                compute AUC for this class
        
        add xlabel='False Positive Rate'
        add ylabel='True Positive Rate (Recall)'
        add title='ROC Curve'
        add legend with AUC scores
        add grid
        
        if save_path:
            save figure
        
        return fig
    """
    pass

def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """
    Plot Precision-Recall curve (useful for imbalanced datasets).
    
    Parameters:
        y_true       : True labels
        y_pred_proba : Predicted probabilities
        save_path    : Path to save figure
    
    Returns:
        fig          : Matplotlib figure
    
    Pseudocode:
        for different probability thresholds (0.0 to 1.0):
            threshold predictions at this value
            compute precision and recall
            store values
        
        plot precision vs recall curve
        
        add xlabel='Recall'
        add ylabel='Precision'
        add title='Precision-Recall Curve'
        
        highlight optimal threshold (highest F1-score)
        
        if save_path:
            save figure
        
        return fig
    """
    pass

def plot_feature_importance(model, feature_names=None, top_k=20, save_path=None):
    """
    Plot feature importance for interpretable models.
    
    Parameters:
        model         : Trained model with feature importance (RandomForest, LogisticRegression)
        feature_names : List of feature names
        top_k         : Number of top features to display
        save_path     : Path to save figure
    
    Returns:
        fig           : Matplotlib figure
    
    Pseudocode:
        extract feature importances from model:
            if RandomForest: use model.feature_importances_
            if LogisticRegression: use abs(model.coef_)
        
        if no feature_names:
            use indices ['Feature 0', 'Feature 1', ...]
        
        sort features by importance (descending)
        select top_k features
        
        create horizontal bar plot:
            x-axis = importance value
            y-axis = feature names
            bars sorted by importance
        
        add xlabel='Importance Score'
        add title=f'Top {top_k} Feature Importances'
        
        if save_path:
            save figure
        
        return fig
    """
    pass

def plot_model_comparison(models_results, metric='accuracy', save_path=None):
    """
    Compare multiple models on a single metric.
    
    Parameters:
        models_results : Dictionary mapping model_name -> metric_value
                        e.g., {'SimpleCNN': 0.85, 'ResNet': 0.92, 'RandomForest': 0.78}
        metric         : Metric name to display
        save_path      : Path to save figure
    
    Returns:
        fig            : Matplotlib figure
    
    Pseudocode:
        extract model names and scores from dictionary
        sort models by score (descending)
        
        create bar plot:
            x-axis = model names
            y-axis = metric scores
            color bars by performance (green=best, red=worst)
        
        add value labels on top of each bar
        add ylabel=metric name (e.g., 'Accuracy')
        add title=f'Model Comparison - {metric}'
        add horizontal line at baseline (e.g., 0.5 for binary classification)
        
        if save_path:
            save figure
        
        return fig
    """
    pass
