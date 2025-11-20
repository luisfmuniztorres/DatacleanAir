import pandas as pd
import numpy as np
from .outliers import hampel_filter, iqr_filter, rate_of_change_filter
from .filters import kalman_filter
from .utils import validate_series

def clean_series_pipeline(
    series: pd.Series,
    apply_hampel: bool = True,
    apply_iqr: bool = False,
    apply_roc: bool = True,
    kalman_Q: float = 0.01,
    kalman_R: float = 1.5
) -> pd.DataFrame:
    """
    High level pipeline: pre-qc -> hampel -> rate-of-change -> kalman -> return df.
    """
    arr = validate_series(series)
    raw = arr.copy()

    # Pre-QC: clamp negatives to NaN
    raw[raw < 0] = np.nan
    # simple forward/backfill for initial NaNs
    if np.isnan(raw).any():
        idx = np.arange(len(raw))
        good = ~np.isnan(raw)
        if good.sum() == 0:
            raise ValueError("Series contains no valid numeric values.")
        raw[np.isnan(raw)] = np.interp(idx[np.isnan(raw)], idx[good], raw[good])

    step = raw.copy()
    if apply_hampel:
        step = hampel_filter(step, window_size=6, n_sigmas=3.0)
    if apply_roc:
        step = rate_of_change_filter(step, max_change=200.0)
    if apply_iqr:
        step = iqr_filter(step, q_low=0.01, q_high=0.99)

    kal = kalman_filter(step, Q=kalman_Q, R=kalman_R)

    return pd.DataFrame({
        "raw": raw,
        "after_hampel": step,
        "kalman": kal
    })
