from typing import Tuple
import numpy as np
import pandas as pd

def validate_series(series: pd.Series) -> np.ndarray:
    """
    Validate input pandas Series: no non-numeric values, drop NaNs.
    Returns numpy float array and raises on invalid types.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")
    arr = pd.to_numeric(series, errors="coerce").values.astype(float)
    # keep NaNs for caller to handle or drop
    return arr

def simple_stats(arr: np.ndarray) -> Tuple[float, float]:
    """Return mean and std (ignoring NaNs)."""
    mean = float(np.nanmean(arr))
    std = float(np.nanstd(arr))
    return mean, std

def z_score_normalize(arr: np.ndarray) -> np.ndarray:
    """
    Z-score normalization: (x - mean) / std , ignoring NaNs.
    """
    x = np.asarray(arr, dtype=float)
    mean = np.nanmean(x)
    std = np.nanstd(x)

    if std == 0 or np.isnan(std):
        return np.zeros_like(x)

    return (x - mean) / std
