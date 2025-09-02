import numpy as np
from typing import List, Dict, Union, Callable

# Define type hints for clarity
BinaryMask = np.ndarray
MetricResult = Union[float, bool]

def _get_stats(y_true: BinaryMask, y_pred: BinaryMask, smooth: float = 1e-6) -> tuple:
    """
    A private helper function to compute the true positives, false positives, 
    and false negatives for binary masks.
    """
    # Ensure inputs are binary arrays of integers (0 or 1)
    y_true = (y_true > 0).astype(int).flatten()
    y_pred = (y_pred > 0).astype(int).flatten()
    
    intersection = np.sum(y_true * y_pred)
    
    tp = intersection
    fp = np.sum(y_pred) - intersection
    fn = np.sum(y_true) - intersection
    
    return tp, fp, fn, smooth

def precision(y_true: BinaryMask, y_pred: BinaryMask, smooth: float = 1e-6) -> float:
    """Calculates precision for binary masks."""
    tp, fp, _, smooth_val = _get_stats(y_true, y_pred, smooth)
    return (tp + smooth_val) / (tp + fp + smooth_val)

def recall(y_true: BinaryMask, y_pred: BinaryMask, smooth: float = 1e-6) -> float:
    """Calculates recall for binary masks."""
    tp, _, fn, smooth_val = _get_stats(y_true, y_pred, smooth)
    return (tp + smooth_val) / (tp + fn + smooth_val)

def f1_score(y_true: BinaryMask, y_pred: BinaryMask, smooth: float = 1e-6) -> float:
    """Calculates the F1-score for binary masks."""
    tp, fp, fn, smooth_val = _get_stats(y_true, y_pred, smooth)
    return (2 * tp + smooth_val) / (2 * tp + fp + fn + smooth_val)

def intersection_over_union(y_true: BinaryMask, y_pred: BinaryMask, smooth: float = 1e-6) -> float:
    """Calculates Intersection over Union (IoU) for binary masks."""
    tp, fp, fn, smooth_val = _get_stats(y_true, y_pred, smooth)
    union = tp + fp + fn
    return (tp + smooth_val) / (union + smooth_val)

def iou_threshold(y_true: BinaryMask, y_pred: BinaryMask, threshold: float = 0.7) -> bool:
    """Checks if the IoU is above a certain threshold."""
    iou = intersection_over_union(y_true, y_pred)
    return iou >= threshold

# --- Metric Registry ---
# This dictionary maps string names to their respective functions,
# allowing for a flexible and extensible interface.

_METRICS_REGISTRY: Dict[str, Callable] = {
    'precision': precision,
    'recall': recall,
    'f1_score': f1_score,
    'iou': intersection_over_union,
    'iou_at_0.7': lambda yt, yp: iou_threshold(yt, yp, threshold=0.7),
}

def get_available_metrics() -> List[str]:
    """Returns a list of all available metric names."""
    return list(_METRICS_REGISTRY.keys())

def compute_metrics(y_true: BinaryMask, y_pred: BinaryMask, metrics: List[str]) -> Dict[str, MetricResult]:
    """
    Computes a list of specified metrics from their string names.

    Args:
        y_true (np.ndarray): The ground truth binary mask.
        y_pred (np.ndarray): The predicted binary mask.
        metrics (List[str]): A list of metric names to compute.
                             See get_available_metrics() for options.

    Returns:
        Dict[str, Union[float, bool]]: A dictionary mapping metric names
                                       to their computed scores.
    
    Raises:
        ValueError: If a requested metric is not in the registry.
    """
    results = {}
    for metric_name in metrics:
        metric_name = metric_name.lower() # make it case-insensitive
        if metric_name not in _METRICS_REGISTRY:
            raise ValueError(
                f"Metric '{metric_name}' not recognized. "
                f"Available metrics: {get_available_metrics()}"
            )
        
        metric_func = _METRICS_REGISTRY[metric_name]
        results[metric_name] = metric_func(y_true, y_pred)
        
    return results