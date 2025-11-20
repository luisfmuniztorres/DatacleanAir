import pandas as pd
import numpy as np

from .outliers import hampel_filter, iqr_filter, rate_of_change_filter
from .filters import kalman_filter
from .utils import validate_series

def clean(
    series: pd.Series,
    apply_hampel: bool = True,
    apply_roc: bool = True,
    apply_iqr: bool = True,
    apply_kalman: bool = True,
    kalman_Q: float = 0.01,
    kalman_R: float = 1.0
) -> pd.DataFrame:
    """
    Full QC/QA cleaning pipeline for PM time series.
    Applies (optionally):
        - Hampel filter
        - Rate of Change (ROC)
        - IQR filter
        - Kalman smoothing
    """

    arr = validate_series(series)
    raw = arr.copy()

    # Pre-QC: clamp negatives
    raw[raw < 0] = np.nan
    if np.isnan(raw).any():
        idx = np.arange(len(raw))
        good = ~np.isnan(raw)
        raw[np.isnan(raw)] = np.interp(idx[np.isnan(raw)], idx[good], raw[good])

    step = raw.copy()

    if apply_hampel:
        step = hampel_filter(step, window_size=6, n_sigmas=3.0)

    if apply_roc:
        step = rate_of_change_filter(step, max_change=200.0)

    if apply_iqr:
        step = iqr_filter(step, q_low=0.01, q_high=0.99)

    if apply_kalman:
        final = kalman_filter(step, Q=kalman_Q, R=kalman_R)
    else:
        final = step

    return pd.DataFrame({
        "raw": raw,
        "after_filters": step,
        "kalman": final
    })
