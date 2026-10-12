"""Two-step cosinor analysis for t response to sinusoidal stimulation.

Step 1 - Low-pass filtering → tonic (sustained) component
==========================================================
A zero-phase Butterworth low-pass filter (default cutoff 0.2 Hz) separates
the slow sustained pupil constriction from the fast 0.4 Hz sinusoidal tracking.

The filter is applied to each participant × condition × block trace (full
trial: pre-stim + stim + post-stim).  The tonic component is characterised
by summary features extracted during the *stimulus* window only:

    tonic_min        minimum of filtered signal during stimulus (peak constriction,
                     negative on baseline-corrected data)
    tonic_min_time   time of that minimum (seconds from stimulus onset)
    tonic_mean       mean of filtered signal during stimulus

Step 2 - Cosinor fit on residual → sinusoidal tracking
=======================================================
The residual (original − low-pass filtered) isolates the fast oscillatory
component.  A cosinor at the fundamental stimulus frequency *f* plus its
2nd harmonic *2f* is fit via OLS on the analysis window of the residual
(controlled by DEFAULT_T_STIM_START, DEFAULT_T_STIM_END, and
DEFAULT_ADD_POSTSTIM_SEC in analysis_io.py):

    y_res(t) = β₀ + α_sin · sin(2πft) + β_cos · cos(2πft)
                   + α_sin_2f · sin(2π·2f·t) + β_cos_2f · cos(2π·2f·t)

Equivalent amplitude-phase form, used consistently across the pupil, saliva
and circadian cosinors of this package:

    y_res(t) = β₀ + A · cos(2πft + φ)

so that φ is the phase angle and the peak occurs at t = -φ / (2πf).

Derived quantities (per harmonic):
    amplitude       = √(α_sin²  + β_cos²)     (fundamental)
    amplitude_2f    = √(α_sin_2f² + β_cos_2f²)    (2nd harmonic)
    amplitude_ratio = amplitude / amplitude_2f (NaN if amplitude_2f = 0)
    phase           = atan2(-α_sin, β_cos)     [rad]
    phase_2f        = atan2(-α_sin_2f, β_cos_2f)   [rad]

Usage
-----
    python -m civibeplr.fit_2step_cosinor_trace_model [--cutoff 0.3] [--no-plot]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import butter, sosfiltfilt

from civibeplr.data_io import (
    load_aggregated_data,
    DEFAULT_PARTICIPANT_COL,
    DEFAULT_CONDITION_COL,
    DEFAULT_BLOCK_COL,
    DEFAULT_TIME_COL,
    DEFAULT_TRIALPHASE_COL,
)
from civibeplr.analysis_io import (
    DEFAULT_T_STIM_START,
    DEFAULT_T_STIM_END,
    DEFAULT_ADD_POSTSTIM_SEC,
)
from civibeplr.latex_table_creation import save_pupil_cosinor_2step_latex

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = DEFAULT_PROJECT_DIR / "results"

DEFAULT_FREQUENCY = 0.4          # Hz - stimulus modulation frequency
DEFAULT_OUTCOME_COL = "rel_size_mm"

DEFAULT_CUTOFF_HZ = 0.2         # low-pass cutoff (well below 0.4 Hz)
DEFAULT_FILTER_ORDER = 2        # Butterworth order (effective 4th with filtfilt)

_CIRCULAR_COLS = {"phase_rad", "phase_2f_rad"}

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Low-pass filtering → tonic component
# ═══════════════════════════════════════════════════════════════════════════

def lowpass_filter(
    signal: np.ndarray,
    fs: float,
    cutoff: float = DEFAULT_CUTOFF_HZ,
    order: int = DEFAULT_FILTER_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter.

    The ``butter``, the Butterworth (from scipy.signal.butter) filter is designed for a maximally flat frequency response in the passband. The use of ``sosfiltfilt``, second-order sections (SOS, from scipy.signal.sosfiltfilt) improves numerical stability for high-order filters. Zero-phase filtering is achieved by applying the filter forward and backward using `sosfiltfilt` (doubles the effective filter order).


    Parameters
    ----------
    signal : array
        Input signal (1-D).
    fs : float
        Sampling frequency (Hz).
    cutoff : float
        Cutoff frequency of the low-pass filter in Hz. Must be less than the Nyquist frequency (fs/2).
    order : int
        Order of the Butterworth filter. Higher order results in a steeper roll-off.

    Returns
    -------
    filtered : array
        Low-pass filtered signal.
    """
    # If the cutoff frequency is greater than or equal to the Nyquist frequency.
    nyq = fs / 2.0
    if cutoff >= nyq:
        raise ValueError(f"Cutoff {cutoff} Hz >= Nyquist {nyq} Hz")
    sos = butter(order, cutoff, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def extract_tonic_and_residual(
    t: np.ndarray,
    y: np.ndarray,
    cutoff: float = DEFAULT_CUTOFF_HZ,
    order: int = DEFAULT_FILTER_ORDER,
    t_stim_start: float = DEFAULT_T_STIM_START,
    t_stim_end: float = DEFAULT_T_STIM_END,
) -> dict:
    """Low-pass filter -> tonic + residual, with tonic features during stim.

    Parameters
    ----------
    t : array
        Time vector (seconds).
    y : array
        Pupil signal (baseline-corrected).
    cutoff : float
        Low-pass cutoff frequency (Hz).
    order : int
        Butterworth filter order.
    t_stim_start, t_stim_end : float
        Stimulus window boundaries (seconds).

    Returns
    -------
    dict with keys:
        tonic           - low-pass filtered signal (full trace)
        residual        - original minus tonic (full trace)
        tonic_min       - minimum of tonic during stimulus (peak constriction)
        tonic_min_time  - time of that minimum (s from stimulus onset)
        tonic_mean      - mean of tonic during stimulus
        fs              - estimated sampling rate (Hz)
    """
    # Estimate sampling rate from median time step
    dt = np.median(np.diff(t))
    fs = 1.0 / dt

    nan_result = {
        "tonic": np.full_like(y, np.nan),
        "residual": np.full_like(y, np.nan),
        "tonic_min": np.nan,
        "tonic_min_time": np.nan,
        "tonic_mean": np.nan,
        "fs": fs,
    }

    # Handle NaN: interpolate for filtering, restore afterwards
    nan_mask = np.isnan(y)
    y_interp = y.copy()
    if nan_mask.any():
        valid_idx = np.where(~nan_mask)[0]
        if len(valid_idx) < 10:
            return nan_result
        y_interp[nan_mask] = np.interp(
            t[nan_mask], t[~nan_mask], y[~nan_mask],
        )

    # Check minimum signal length for filter (sosfiltfilt needs padding)
    min_length = 3 * (2 * order + 1) * 3
    if len(y_interp) < min_length:
        return nan_result

    # Apply low-pass filter to full trace
    tonic = lowpass_filter(y_interp, fs=fs, cutoff=cutoff, order=order)
    residual = y_interp - tonic

    # Restore NaN positions
    if nan_mask.any():
        tonic[nan_mask] = np.nan
        residual[nan_mask] = np.nan

    # Extract tonic features during stimulus window
    stim_mask = (t >= t_stim_start) & (t <= t_stim_end) & ~nan_mask
    if stim_mask.any():
        tonic_stim = tonic[stim_mask]
        t_stim = t[stim_mask]
        tonic_min = float(np.nanmin(tonic_stim))
        tonic_min_time = float(t_stim[np.nanargmin(tonic_stim)])
        tonic_mean = float(np.nanmean(tonic_stim))
    else:
        tonic_min = np.nan
        tonic_min_time = np.nan
        tonic_mean = np.nan

    return {
        "tonic": tonic,
        "residual": residual,
        "tonic_min": tonic_min,
        "tonic_min_time": tonic_min_time,
        "tonic_mean": tonic_mean,
        "fs": fs,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Cosinor on residual
# ═══════════════════════════════════════════════════════════════════════════

def _nan_cosinor_result() -> dict[str, float]:
    """Return NaN result dict for failed or insufficient cosinor fits."""
    return {k: np.nan for k in [
        "beta_intercept", "alpha_sin", "beta_cos",
        "alpha_sin_ci_lo", "alpha_sin_ci_hi",
        "beta_cos_ci_lo", "beta_cos_ci_hi",
        "amplitude", "phase_rad",
        "alpha_sin_2f", "beta_cos_2f",
        "alpha_sin_2f_ci_lo", "alpha_sin_2f_ci_hi",
        "beta_cos_2f_ci_lo", "beta_cos_2f_ci_hi",
        "amplitude_2f", "phase_2f_rad",
        "amplitude_ratio",
        "r_squared", "adj_r_squared", "aic", "bic", "rmse", "n_obs",
    ]}


def fit_cosinor_residual(
    t: np.ndarray,
    y_residual: np.ndarray,
    freq: float = DEFAULT_FREQUENCY,
    t_stim_start: float = DEFAULT_T_STIM_START,
    t_stim_end: float = DEFAULT_T_STIM_END,
) -> dict[str, float]:
    """Fit cosinor (fundamental + 2nd harmonic) on residual during stim.

    The design matrix contains the fundamental frequency *f* and the 2nd
    harmonic *2f*:

        y = β₀ + α_sin·sin(ωt) + β_cos·cos(ωt)
                + α_sin_2f·sin(2ωt) + β_cos_2f·cos(2ωt)

    Parameters
    ----------
    t : array
        Time vector (full trace).
    y_residual : array
        Residual signal (original − tonic, full trace).
    freq : float
        Stimulus frequency (Hz).
    t_stim_start, t_stim_end : float
        Stimulus window boundaries.

    Returns
    -------
    dict with β₀, α_sin, β_cos, α_sin_2f, β_cos_2f, amplitudes, phases,
    CIs, and fit statistics.
    """
    # Select stimulus window
    stim_mask = (t >= t_stim_start) & (t <= t_stim_end)
    t_stim = t[stim_mask]
    y_stim = y_residual[stim_mask]

    # Remove NaN
    valid = ~np.isnan(y_stim)
    t_clean = t_stim[valid]
    y_clean = y_stim[valid]

    if len(y_clean) < 7:  # need > number of regressors (5) + 1
        return _nan_cosinor_result()

    # Design matrix: [1, sin(ωt), cos(ωt), sin(2ωt), cos(2ωt)]
    omega = 2.0 * np.pi * freq
    X = np.column_stack([
        np.ones(len(t_clean)),
        np.sin(omega * t_clean),
        np.cos(omega * t_clean),
        np.sin(2.0 * omega * t_clean),
        np.cos(2.0 * omega * t_clean),
    ])

    try:
        model = sm.OLS(y_clean, X)
        results = model.fit()
    except (np.linalg.LinAlgError, ValueError):
        return _nan_cosinor_result()

    b = results.params
    ci = results.conf_int(alpha=0.05)

    # Convention: y = β₀ + A·cos(2πft + φ), so φ = atan2(-α_sin, β_cos)
    # and the peak occurs at t = -φ / (2πf).

    # Fundamental (f)
    amplitude = np.sqrt(b[1] ** 2 + b[2] ** 2)
    phase_rad = np.arctan2(-b[1], b[2])

    # 2nd harmonic (2f)
    amplitude_2f = np.sqrt(b[3] ** 2 + b[4] ** 2)
    phase_2f_rad = np.arctan2(-b[3], b[4])

    return {
        "beta_intercept": b[0],
        "alpha_sin": b[1],
        "beta_cos": b[2],
        "alpha_sin_ci_lo": ci[1, 0],
        "alpha_sin_ci_hi": ci[1, 1],
        "beta_cos_ci_lo": ci[2, 0],
        "beta_cos_ci_hi": ci[2, 1],
        "amplitude": amplitude,
        "phase_rad": phase_rad,
        "alpha_sin_2f": b[3],
        "beta_cos_2f": b[4],
        "alpha_sin_2f_ci_lo": ci[3, 0],
        "alpha_sin_2f_ci_hi": ci[3, 1],
        "beta_cos_2f_ci_lo": ci[4, 0],
        "beta_cos_2f_ci_hi": ci[4, 1],
        "amplitude_2f": amplitude_2f,
        "phase_2f_rad": phase_2f_rad,
        "amplitude_ratio": amplitude / amplitude_2f if amplitude_2f != 0 else np.nan,
        "r_squared": results.rsquared,
        "adj_r_squared": results.rsquared_adj,
        "aic": results.aic,
        "bic": results.bic,
        "rmse": np.sqrt(results.mse_resid),
        "n_obs": len(y_clean),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def fit_all_groups(
    df: pd.DataFrame,
    outcome_col: str = DEFAULT_OUTCOME_COL,
    time_col: str = DEFAULT_TIME_COL,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    phase_col: str = DEFAULT_TRIALPHASE_COL,
    freq: float = DEFAULT_FREQUENCY,
    cutoff: float = DEFAULT_CUTOFF_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
    t_stim_start: float = DEFAULT_T_STIM_START,
    t_stim_end: float = DEFAULT_T_STIM_END,
    add_poststim_sec: float = DEFAULT_ADD_POSTSTIM_SEC,
) -> pd.DataFrame:
    """Run two-step analysis for all participant × condition × block units.

    Parameters
    ----------
    df : DataFrame
        Aggregated pupil data.
    outcome_col : str
        Baseline-corrected outcome column.
    freq : float
        Stimulus modulation frequency (Hz).
    cutoff : float
        Low-pass filter cutoff (Hz).
    filter_order : int
        Butterworth filter order.
    t_stim_start, t_stim_end : float
        Stimulus window boundaries (seconds).

    Returns
    -------
    DataFrame with one row per participant × condition × block, containing
    tonic features (Step 1) and cosinor parameters (Step 2).
    """
    grouping_cols = [participant_col, condition_col, block_col]
    rows: list[dict] = []
    conditions = sorted(df[condition_col].dropna().unique())

    # groups representing participant × condition × block data
    for group_key, group_data in df.groupby(grouping_cols, dropna=False):
        group_data = group_data.sort_values(time_col)
        t = group_data[time_col].values.astype(float)
        y = group_data[outcome_col].values.astype(float)

        # Step 1: low-pass filter → tonic component
        tonic_out = extract_tonic_and_residual(
            t, y, cutoff=cutoff, order=filter_order,
            t_stim_start=t_stim_start, t_stim_end=t_stim_end,
        )

        # Step 2: cosinor on residual — fit window is stimulus window (+ add_poststim_sec if non-zero)
        cosinor_out = fit_cosinor_residual(
            t,
            tonic_out["residual"],
            freq=freq,
            t_stim_start=t_stim_start,
            t_stim_end=(t_stim_end + add_poststim_sec),
        )

        # Build result row
        row: dict = {}
        for i, col in enumerate(grouping_cols):
            row[col] = group_key[i] if isinstance(group_key, tuple) else group_key

        # Baseline: mean of outcome during pre-stim phase
        pre_stim = group_data[group_data[phase_col] == "pre-stim"]["pupil_mean"]
        row["baseline_mean"] = float(pre_stim.mean()) if len(pre_stim) > 0 else np.nan

        row["tonic_min"] = tonic_out["tonic_min"]
        row["tonic_min_time"] = tonic_out["tonic_min_time"]
        row["tonic_mean"] = tonic_out["tonic_mean"]
        row.update(cosinor_out)
        rows.append(row)

    # Progress summary per condition
    result_df = pd.DataFrame(rows)
    for cond in conditions:
        cond_rows = result_df[result_df[condition_col] == cond]
        n_valid = cond_rows["r_squared"].notna().sum()
        print(f"  {cond}: {n_valid}/{len(cond_rows)} valid fits")

    return result_df


def main(
    input_df: pd.DataFrame,
    outcome_col: str = DEFAULT_OUTCOME_COL,
    time_col: str = DEFAULT_TIME_COL,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    freq: float = DEFAULT_FREQUENCY,
    cutoff: float = DEFAULT_CUTOFF_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
    t_stim_start: float = DEFAULT_T_STIM_START,
    t_stim_end: float = DEFAULT_T_STIM_END,
    add_poststim_sec: float = DEFAULT_ADD_POSTSTIM_SEC,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run the full two-step pipeline and save results CSV.

    Returns
    -------
    results_df : DataFrame
        One row per participant × condition × block with all parameters.
    """
    if output_dir is None:
        output_dir = DEFAULT_RESULTS_DIR
    results_dir = output_dir / "tables" / "pupil_cosinor"

    # ── Validate stim window against "stim" trial phase ───────────────────
    stim_rows = input_df[input_df[DEFAULT_TRIALPHASE_COL] == "stim"]
    if stim_rows.empty:
        raise ValueError(
            f"No rows with {DEFAULT_TRIALPHASE_COL}='stim' found in data."
        )
    stim_t_min = stim_rows[time_col].min()
    stim_t_max = stim_rows[time_col].max()
    if t_stim_start < stim_t_min - 0.05 or t_stim_end > stim_t_max + 0.05:
        raise ValueError(
            f"Selected stim window [{t_stim_start}, {t_stim_end}] s exceeds "
            f"the 'stim' trial-phase range [{stim_t_min:.3f}, {stim_t_max:.3f}] s."
        )
    # print(f"  ✓ Stim window [{t_stim_start}, {t_stim_end}] s is consistent "
    #       f"with trial phase 'stim' [{stim_t_min:.3f}, {stim_t_max:.3f}] s")

    # ── Fit all groups ────────────────────────────────────────────────────
    print("=" * 65)
    print("TWO-STEP COSINOR ANALYSIS")
    print(f"  Low-pass cutoff: {cutoff} Hz  |  Filter order: {filter_order}")
    fit_end = t_stim_end + add_poststim_sec
    if add_poststim_sec > 0:
        print(f"  Cosinor frequency: {freq} Hz  |  Stim window: {t_stim_start}-{t_stim_end} s  |  Fit window: {t_stim_start}-{fit_end} s (+{add_poststim_sec} s post-stim buffer)")
    else:
        print(f"  Cosinor frequency: {freq} Hz  |  Stim window: {t_stim_start}-{t_stim_end} s  |  Fit window: stimulus window only")
    print("=" * 65)

    results_2step_param_df = fit_all_groups(
        df=input_df, outcome_col=outcome_col, time_col=time_col,
        participant_col=participant_col, condition_col=condition_col,
        block_col=block_col, freq=freq, cutoff=cutoff,
        filter_order=filter_order,
        t_stim_start=t_stim_start, t_stim_end=t_stim_end,
        add_poststim_sec=add_poststim_sec,
    )

    # Diagnostics over all raw fits (participant x condition x block). Median is used
    # to reduce skew from outlier fits. These are not the reported grand averages --
    # those are computed hierarchically in save_pupil_cosinor_2step_latex (mean across
    # blocks -> mean across conditions -> mean across participants), giving each
    # participant equal weight.
    valid = results_2step_param_df["r_squared"].notna()
    print(f"\nTotal: {len(results_2step_param_df)} fits  |  Valid: {valid.sum()}")
    print(f"Median R² (cosinor on residual): {results_2step_param_df.loc[valid, 'r_squared'].median():.4f}")
    print(f"Median tonic min: {results_2step_param_df.loc[valid, 'tonic_min'].median():.4f} mm after {results_2step_param_df.loc[valid, 'tonic_min_time'].median():.3f} sec.")
    print(f"Median sinusoidal amplitude: {results_2step_param_df.loc[valid, 'amplitude'].median():.4f} mm")

    # ── Save settings to JSON if requested ─────────────────────────────────
    save_settings_json = DEFAULT_RESULTS_DIR / "metadata" / "pupil_cosinor_2step_settings.json"
    save_settings_json.parent.mkdir(parents=False, exist_ok=True)

    if save_settings_json is not None:
        settings = {
            "cutoff": cutoff,
            "filter_order": filter_order,
            "freq": freq,
            "t_stim_start": t_stim_start,
            "t_stim_end": t_stim_end,
            "add_poststim_sec": add_poststim_sec,
            "outcome_col": outcome_col,
        }
        with open(save_settings_json, "w") as f:
            json.dump(settings, f, indent=2)
        print(f"\nSaved decomposition settings to {save_settings_json}")

    # ── Save results ──────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "pupil_cosinor_2step_param.csv"
    results_2step_param_df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    save_pupil_cosinor_2step_latex(results_df=results_2step_param_df, output_dir=output_dir)

    return results_2step_param_df


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Two-step cosinor analysis: low-pass → tonic, cosinor → residual.",
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to aggregated data CSV (default: data/aggregated/data.csv).",
    )
    parser.add_argument(
        "--outcome", type=str, default=DEFAULT_OUTCOME_COL,
        help=f"Outcome column (default: {DEFAULT_OUTCOME_COL}).",
    )
    parser.add_argument(
        "--freq", type=float, default=DEFAULT_FREQUENCY,
        help=f"Stimulus frequency in Hz (default: {DEFAULT_FREQUENCY}).",
    )
    parser.add_argument(
        "--cutoff", type=float, default=DEFAULT_CUTOFF_HZ,
        help=f"Low-pass cutoff in Hz (default: {DEFAULT_CUTOFF_HZ}).",
    )
    parser.add_argument(
        "--filter-order", type=int, default=DEFAULT_FILTER_ORDER,
        help=f"Butterworth filter order (default: {DEFAULT_FILTER_ORDER}).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory for results CSV.",
    )
    parser.add_argument(
        "--add-poststim-sec", type=float, default=DEFAULT_ADD_POSTSTIM_SEC,
        help=f"Extra post-stimulus seconds included in the cosinor fit window (default: {DEFAULT_ADD_POSTSTIM_SEC}).",
    )
    args = parser.parse_args()

    df = load_aggregated_data(args.data)
    main(
        df,
        outcome_col=args.outcome,
        freq=args.freq,
        cutoff=args.cutoff,
        filter_order=args.filter_order,
        add_poststim_sec=args.add_poststim_sec,
        output_dir=args.output_dir,
    )
