"""One explicit Gaussian likelihood for every case and selection method."""
import numpy as np
from .preprocessing import validate_weights

def information_criteria(y, prediction, effective_df, *, weights=None):
    """Conditional Gaussian IC, Var(error_i)=sigma²/w_i, with fitted sigma².

    Include the variance parameter and log determinant of the weights. For
    penalized/constrained fits df is a local effective dimension. IC remains a
    conditional approximation after data-dependent sparse model selection.
    """
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.shape != prediction.shape or y.ndim != 1 or not len(y):
        raise ValueError("IC requires matching nonempty observation vectors")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(prediction)) or effective_df < 0:
        raise ValueError("IC inputs must be finite with nonnegative effective dimension")
    w = validate_weights(weights, len(y))
    n = len(y)
    scale = max(float(np.mean(w * y ** 2)), np.finfo(float).tiny ** 0.5)
    variance = max(float(np.sum(w * (y - prediction) ** 2) / n), np.finfo(float).eps * scale)
    minus_two_log_likelihood = n * (np.log(2 * np.pi * variance) + 1) - np.sum(np.log(w))
    k = effective_df + 1
    return float(minus_two_log_likelihood + 2 * k), float(minus_two_log_likelihood + np.log(n) * k)
