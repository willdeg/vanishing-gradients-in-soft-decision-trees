from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    log_loss: float


def classification_metrics(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    if y_prob is None:
        log_loss = float("nan")
    else:
        y_prob = np.clip(np.asarray(y_prob), 1e-12, 1 - 1e-12)
        log_loss = -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))

    return BinaryMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        log_loss=float(log_loss),
    )


def metrics_to_frame(rows):
    records = []
    for model_name, metrics in rows:
        records.append(
            {
                "Model": model_name,
                "Accuracy": metrics.accuracy,
                "Precision": metrics.precision,
                "Recall": metrics.recall,
                "F1": metrics.f1,
                "LogLoss": metrics.log_loss,
            }
        )
    return pd.DataFrame(records)
