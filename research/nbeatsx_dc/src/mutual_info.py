"""
mutual_info.py
Histogram-based mutual information and temporal mutual information (TMI).
"""

import numpy as np


def _hist_mi(x: np.ndarray, y: np.ndarray, bins: int = 20) -> float:
    """Estimate mutual information via joint histogram."""
    c_xy, _, _ = np.histogram2d(x, y, bins=bins)
    c_xy = c_xy + 1e-10  # Laplace smoothing
    px = c_xy.sum(axis=1)
    py = c_xy.sum(axis=0)
    pxy = c_xy / c_xy.sum()
    px /= px.sum()
    py /= py.sum()
    # I(X;Y) = Σ p(x,y) log[p(x,y)/(p(x)p(y))]
    outer = np.outer(px, py) + 1e-10
    mi = (pxy * np.log(pxy / outer + 1e-10)).sum()
    return float(max(0.0, mi))


def temporal_mi(
    x: np.ndarray,
    y: np.ndarray,
    tau_max: int = 0,
    bins: int = 20,
) -> float:
    """
    Temporal mutual information: max MI over lags 0..tau_max.
    tau > 0 tests whether x(t) predicts y(t+tau) or vice versa.
    """
    if tau_max == 0:
        return _hist_mi(x, y, bins)
    best = 0.0
    n = len(x)
    for tau in range(0, tau_max + 1):
        if tau == 0:
            mi = _hist_mi(x, y, bins)
        else:
            # x leads y
            mi = max(
                _hist_mi(x[:n - tau], y[tau:], bins),
                _hist_mi(y[:n - tau], x[tau:], bins),
            )
        best = max(best, mi)
    return best
