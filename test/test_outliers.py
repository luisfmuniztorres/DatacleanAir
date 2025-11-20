import numpy as np
from DatacleanAir.outliers import hampel_filter

def test_hampel_detects_spike():
    x = np.ones(51)
    x[25] = 100.0
    y = hampel_filter(x, window_size=5, n_sigmas=3.0)
    assert y[25] != 100.0
