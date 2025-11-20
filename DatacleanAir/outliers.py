import numpy as np
from scipy.stats import median_abs_deviation
from typing import Optional

def hampel_filter(data: np.ndarray, window_size: int = 6, n_sigmas: float = 3.0) -> np.ndarray:
    """
    Hampel filter: replace detected outliers with local median.
    window_size: half-window size (so actual window length = 2*window_size)
    """
    x = np.asarray(data, dtype=float).copy()
    n = len(x)
    if n == 0:
        return x
    for i in range(window_size, n - window_size):
        window = x[i - window_size : i + window_size + 1]
        med = np.nanmedian(window)
        mad = median_abs_deviation(window, nan_policy="omit")
        if mad == 0 or np.isnan(mad):
            continue
        threshold = n_sigmas * mad
        if np.abs(x[i] - med) > threshold:
            x[i] = med
    return x

def iqr_filter(data: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> np.ndarray:
    """
    Clip values outside percentile bounds.
    """
    low = np.nanquantile(data, q_low)
    high = np.nanquantile(data, q_high)
    out = np.asarray(data, dtype=float).copy()
    out[out < low] = low
    out[out > high] = high
    return out

def rate_of_change_filter(data: np.ndarray, max_change: float = 200.0) -> np.ndarray:
    """
    Replace points that change more than max_change per step with NaN (or linear interp).
    """
    x = np.asarray(data, dtype=float).copy()
    diffs = np.abs(np.diff(x))
    bad = np.where(diffs > max_change)[0] + 1
    x[bad] = np.nan
    # simple linear interpolation of NaNs
    nans = np.isnan(x)
    if np.any(nans):
        idx = np.arange(len(x))
        good = ~nans
        x[nans] = np.interp(idx[nans], idx[good], x[good])
    return x
