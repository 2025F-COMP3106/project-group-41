#Evaluate metrics for model performance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
#Accuracy score for % of correct pridictions (correct/total)
#percision score for predicted positives how many were acctually positive (True Positive / (True Positive + False Negative))
#recall score for the actual positives how many where found (True Positive / (True Positive + False Negative))
#f1 score harmonic mean of percision & recall
#confusion matrix a 2x2 table showing true positives, true negatives, false positives, false negatives and counting the total for each cell

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics
    Parameters
        y_true: True Labels(n samples)
        y_pred: Predicted labels (n samples)
    Returns:
        metrics_dict: Dictionary containing all metrics
    """
    #Compute metrics with sklearn
    accuracy = accuracy_score(y_true, y_pred)                                          #correct / total
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)   #TP / (TP + FP)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)         #TP / (TP + FN)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)                 #harmonic mean

    #Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    #metrics dictionary
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

    return metrics_dict