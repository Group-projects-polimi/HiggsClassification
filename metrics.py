"""Metrics for models."""

import numpy as np

from numpy.typing import NDArray


def f1_score(y_true: NDArray[np.int16], y_pred: NDArray[np.int16]) -> float:
    """
    Compute F1 score

    Args:
        y_true: shape=(N,), true labels (0 or 1)
        y_pred: shape=(N,), predicted labels (0 or 1)

    Returns:
        scalar, F1 score
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
