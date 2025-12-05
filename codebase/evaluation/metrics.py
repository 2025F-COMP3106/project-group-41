# Evaluation metrics for model performance

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics
    
    Parameters:
        y_true       : True labels (n_samples,)
        y_pred       : Predicted labels (n_samples,)
    
    Returns:
        metrics_dict : Dictionary with all metrics
    """
    #compute metrics (sklearn)
    accuracy = accuracy_score(y_true, y_pred)                                          #correct / total
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)   #TP / (TP + FP)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)         #TP / (TP + FN)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)                 #harmonic mean
    
    #confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #init metrics dict
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }
    
    return metrics_dict
