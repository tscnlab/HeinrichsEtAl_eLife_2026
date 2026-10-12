"""General utility functions for the civibeplr package."""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtransform_log_amplitude_circadian(
    intercept_samples: np.ndarray,
    alpha_samples: np.ndarray,
    beta_samples: np.ndarray,
    hdi_prob: float = 0.95,
) -> tuple[float, float, float]:
    """Back-transform a log-scale circadian amplitude to original (mm) units.

    The circadian model is:
        log(y) = mu + A_log * cos(2*pi*t/tau + phi) + ...
    where A_log = sqrt(alpha^2 + beta^2) is the amplitude on the log scale,
    phi = atan2(-alpha, beta) is the phase angle (peak at t = -phi/omega),
    and mu is the fixed-effect intercept (= log of the geometric mean response).

    The peak prediction is exp(mu + A_log) and the trough is exp(mu - A_log),
    so the conventional amplitude in mm (half the peak-to-trough range) is:

        A_mm = exp(mu) * sinh(A_log)

    This equals exp(mu) * (exp(A_log) - exp(-A_log)) / 2, which is the exact
    half-range of the back-transformed sinusoid.  Note that A_mm depends on
    the baseline level exp(mu): the same log-scale modulation corresponds to
    a larger mm amplitude when the baseline pupil size is larger.

    Parameters
    ----------
    intercept_samples : 1-D array
        Posterior samples of the fixed-effect intercept (on log scale).
    alpha_samples : 1-D array
        Posterior samples of the sine slope (α).
    beta_samples : 1-D array
        Posterior samples of the cosine slope (β).
    hdi_prob : float
        Credible interval probability (default 0.95).

    Returns
    -------
    (median, lo, hi) : tuple of float
        Posterior median and HDI bounds of A_mm in the original outcome units.
    """
    try:
        import arviz as az
        a_log = np.sqrt(alpha_samples ** 2 + beta_samples ** 2)
        a_mm  = np.exp(intercept_samples) * np.sinh(a_log)
        median = float(np.median(a_mm))
        hdi    = az.hdi(a_mm, hdi_prob=hdi_prob)
        return median, float(hdi[0]), float(hdi[1])
    except Exception:
        return np.nan, np.nan, np.nan


def format_number(x: float, decimals: int = 3) -> str:
    """Format number with smart formatting for display.
    
    Uses fixed decimal places for numbers >= 0.01, otherwise scientific notation.
    This is useful for formatting tables where values span multiple orders of magnitude.
    
    Args:
        x: Number to format.
        decimals: Number of decimal places to use (default: 3).
        
    Returns:
        Formatted string representation of the number.
        
    Examples:
        >>> format_number(0.123)
        '0.123'
        >>> format_number(0.00456)
        '4.560e-03'
        >>> format_number(0.0)
        '0.000'
    """
    if pd.isna(x):
        return ""
    if abs(x) >= 0.01 or x == 0:
        return f"{x:.{decimals}f}"
    else:
        # Scientific notation with mantissa rounded to decimal places
        return f"{x:.{decimals}e}"


# ── Circular statistics for phase angles ─────────────────────────────────────
# Phases live on (-pi, pi], so arithmetic means and SDs break down when angles
# straddle the +/-pi boundary (e.g. -3.0 and +3.0 average to 0 instead of pi).


def _finite_angles(angles) -> np.ndarray:
    a = np.asarray(angles, dtype=float).ravel()
    return a[np.isfinite(a)]


def circ_mean(angles) -> float:
    """Circular mean of angles in radians, in (-pi, pi]; NaNs are ignored."""
    a = _finite_angles(angles)
    if a.size == 0:
        return np.nan
    return float(np.arctan2(np.sin(a).mean(), np.cos(a).mean()))


def circ_sd(angles) -> float:
    """Circular SD sqrt(-2 ln R) in radians, where R is the mean resultant length.

    Approximately equal to the linear SD for tightly clustered angles.
    Returns inf when the directions cancel (R ~ 0). NaNs are ignored.
    """
    a = _finite_angles(angles)
    if a.size == 0:
        return np.nan
    R = np.hypot(np.sin(a).mean(), np.cos(a).mean())
    if R < 1e-12:
        # Directions cancel (up to rounding): mean undefined, spread unbounded
        return np.inf
    # max(..., 0) avoids -0.0 when all angles coincide (R == 1)
    return float(np.sqrt(max(0.0, -2.0 * np.log(min(R, 1.0)))))


def circ_interval(angles, prob: float = 0.95) -> tuple[float, float]:
    """Equal-tailed interval of angles unwrapped around their circular mean.

    Each angle is shifted by a multiple of 2*pi to lie within pi of the circular
    mean before the percentiles are taken, so the bounds may fall outside
    (-pi, pi]. NaNs are ignored.
    """
    a = _finite_angles(angles)
    if a.size == 0:
        return np.nan, np.nan
    m = circ_mean(a)
    unwrapped = m + np.angle(np.exp(1j * (a - m)))
    q = [(1 - prob) / 2 * 100, (1 + prob) / 2 * 100]
    lo, hi = np.percentile(unwrapped, q)
    return float(lo), float(hi)


# ── Private helpers ──────────────────────────────────────────────────────────


def _circ_mean(angles: pd.Series) -> float:
    """Circular (directional) mean of angles in radians."""
    return float(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()))


def _circ_sem(angles: pd.Series) -> float:
    """Circular SEM: SEM ≈ √(V / n) where V = 1 − R̄ (circular variance)."""
    n = len(angles)
    if n < 2:
        return np.nan
    R = np.sqrt(np.sin(angles).mean() ** 2 + np.cos(angles).mean() ** 2)
    return float(np.sqrt((1.0 - R) / n))

