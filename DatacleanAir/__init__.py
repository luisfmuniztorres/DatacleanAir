"""airqc - air quality QC/QA and filtering utilities."""

from .filters import kalman_filter, savgol_filter_wrapper
from .outliers import hampel_filter, iqr_filter, rate_of_change_filter
from .qc import clean_series_pipeline

__all__ = [
    "kalman_filter",
    "savgol_filter_wrapper",
    "hampel_filter",
    "iqr_filter",
    "rate_of_change_filter",
    "clean_series_pipeline",
]
__version__ = "0.1.0"