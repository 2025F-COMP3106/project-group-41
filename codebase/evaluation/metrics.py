import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
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

def calculate_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Generate confusion matrix with labels
    
    Parameters:
        y_true      : True labels
        y_pred      : Predicted labels
        class_names : List of class names for labeling
    
    Returns:
        cm          : Confusion matrix (n_classes, n_classes)
    """
    #confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #if class names provided, return as dict for clarity
    if class_names is not None:
        #create labeled version (dict mapping class pairs to counts)
        labeled_cm = {}                                         #init empty dict
        for i, true_class in enumerate(class_names):            #go through true class labels
            for j, pred_class in enumerate(class_names):        #go through predicted class labels
                #map (true_class, pred_class) tuple to confusion count
                #example: ('benign', 'malignant') -> 15 means 15 benign samples predicted as malignant
                labeled_cm[(true_class, pred_class)] = cm[i, j]
        return cm, labeled_cm                                   #both numeric matrix & labeled dict
    
    return cm                                                   #numeric matrix if no class names

def calculate_per_class_metrics(y_true, y_pred, class_names=None):
    """
    Calculate metrics for each class individually
    
    Parameters:
        y_true      : True labels
        y_pred      : Predicted labels
        class_names : List of class names
    
    Returns:
        per_class   : Dictionary mapping class_name -> metrics
    """
    #compute per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)  #precision for each class
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)        #recall for each class
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)                #f1 for each class
    
    #compute number of samples per class in y_true
    unique_classes, support = np.unique(y_true, return_counts=True)             #count samples per class
    
    #create per-class dictionary
    per_class = {}
    for i, class_label in enumerate(unique_classes):
        #use class name if provided, otherwise use numeric label
        key = class_names[class_label] if class_names is not None else f'class_{class_label}'
        
        #store metrics for this class
        per_class[key] = {
            'precision': precision[i],      #what % of predictions for this class were correct
            'recall': recall[i],            #what % of true samples were found
            'f1_score': f1[i],              #harmonic mean of precision & recall
            'support': support[i]           #number of true samples in dataset
        }
    
    return per_class
