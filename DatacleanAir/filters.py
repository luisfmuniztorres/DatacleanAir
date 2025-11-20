import numpy as np
from scipy.signal import savgol_filter
from typing import Optional

def kalman_filter(data: np.ndarray, Q: float = 0.01, R: float = 1.0, P_init: float = 1.0) -> np.ndarray:
    """
    1D Kalman filter for smoothing time series.
    """
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        return x
    x_est = x[0]
    P = float(P_init)
    out = np.empty_like(x)
    out[0] = x_est
    for k in range(1, x.size):
        z = x[k]
        x_pred = x_est
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x_est = x_pred + K * (z - x_pred)
        P = (1.0 - K) * P_pred
        out[k] = x_est
    return out

def savgol_filter_wrapper(data: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """
    Wrapper for Savitzky-Golay. Ensures window length odd and <= len(data).
    """
    x = np.asarray(data, dtype=float)
    n = x.size
    if n == 0:
        return x
    wl = min(window_length, n if n % 2 == 1 else n - 1)
    if wl < polyorder + 2:
        wl = polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3
    if wl >= n:
        return x
    return savgol_filter(x, window_length=wl, polyorder=polyorder)
