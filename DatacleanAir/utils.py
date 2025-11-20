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
