import numpy as np
from DatacleanAir.filters import kalman_filter

def test_kalman_constant():
    x = np.ones(100) * 5.0
    out = kalman_filter(x, Q=1e-6, R=1e-6, P_init=1e-3)
    assert np.allclose(out, x, rtol=1e-3)
