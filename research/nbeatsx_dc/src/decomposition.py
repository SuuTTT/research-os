"""
decomposition.py
Dual-stage spectral decomposition for NBEATSx-DC.

Stage 1: Remove top-K1 intrinsic frequencies from the target variable.
Stage 2: Extract top-K2 driving frequency components per variable.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FreqComponent:
    var_id: int          # variable index (0 = target residual, 1..n = exogenous)
    freq_idx: int        # FFT bin index
    amplitude: float
    energy: float        # normalised spectral energy within this variable's K2 components
    period: float        # approximate period in samples
    component: np.ndarray  # reconstructed single-frequency time series


def _rfft_top_k(series: np.ndarray, K: int, exclude_dc: bool = True):
    """Return top-K frequency indices sorted by amplitude (descending)."""
    n = len(series)
    fft = np.fft.rfft(series)
    amp = np.abs(fft)
    start = 1 if exclude_dc else 0
    # top K among [start, ..., N//2]
    ranked = np.argsort(amp[start:])[::-1][:K] + start
    return fft, amp, ranked


def _reconstruct(fft: np.ndarray, keep_indices: np.ndarray, n: int) -> np.ndarray:
    """Reconstruct signal keeping only the specified FFT bins."""
    fft_masked = np.zeros_like(fft)
    fft_masked[keep_indices] = fft[keep_indices]
    return np.fft.irfft(fft_masked, n=n)


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def remove_intrinsic_frequencies(
    target: np.ndarray,
    K1: int,
    exclude_dc: bool = True,
) -> tuple:
    """
    Remove the top-K1 dominant (intrinsic) frequencies from *target*.

    Returns
    -------
    residual       : np.ndarray  target minus intrinsic components
    intrinsic      : np.ndarray  reconstructed intrinsic signal
    intrinsic_idx  : np.ndarray  FFT bin indices that were removed
    """
    n = len(target)
    fft, amp, top_idx = _rfft_top_k(target, K1, exclude_dc=exclude_dc)
    intrinsic = _reconstruct(fft, top_idx, n)
    residual = target - intrinsic
    return residual, intrinsic, top_idx


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def extract_freq_components(
    series: np.ndarray,
    var_id: int,
    K2: int,
    exclude_dc: bool = True,
) -> List[FreqComponent]:
    """
    Extract top-K2 frequency components for *series* (one variable).

    Returns a list of FreqComponent objects.
    """
    n = len(series)
    fft, amp, top_idx = _rfft_top_k(series, K2, exclude_dc=exclude_dc)

    energies_raw = amp[top_idx] ** 2
    total = energies_raw.sum() + 1e-8
    energies_norm = energies_raw / total

    components = []
    for rank, idx in enumerate(top_idx):
        comp = _reconstruct(fft, np.array([idx]), n)
        period = n / idx if idx > 0 else float("inf")
        components.append(FreqComponent(
            var_id=var_id,
            freq_idx=int(idx),
            amplitude=float(amp[idx]),
            energy=float(energies_norm[rank]),
            period=period,
            component=comp,
        ))
    return components


def build_all_nodes(
    target_residual: np.ndarray,
    exog: np.ndarray,        # shape (T, n_exog)
    K2: int,
    exclude_dc: bool = True,
) -> List[FreqComponent]:
    """
    Run stage-2 extraction on the residual target (var_id=0) and all
    exogenous variables (var_id=1..n_exog).

    Returns a flat list of FreqComponent nodes.
    """
    nodes: List[FreqComponent] = []
    # var 0 = residual target
    nodes.extend(extract_freq_components(target_residual, var_id=0, K2=K2, exclude_dc=exclude_dc))
    # var 1..n
    for i in range(exog.shape[1]):
        nodes.extend(extract_freq_components(exog[:, i], var_id=i + 1, K2=K2, exclude_dc=exclude_dc))
    return nodes
