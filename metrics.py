from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
)
import numpy as np


def metrics_from_preds(y_true, y_pred, n_classes=4):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)
    eps = 1e-8
    specificity_macro = np.mean(tn / (tn + fp + eps))
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity_macro": float(specificity_macro),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": cm.tolist()
    }
    return out
