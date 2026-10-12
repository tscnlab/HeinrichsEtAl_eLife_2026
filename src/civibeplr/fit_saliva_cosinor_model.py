"""Fit two-harmonic sine-cosine (cosinor) models to clean saliva data.

Fits a linear model per participant including the fundamental and 2nd harmonic:
    y(t) = β₀
         + α_sin  · sin(2πt/τ)  + β_cos  · cos(2πt/τ)      (fundamental)
         + α_sin_2f · sin(4πt/τ)  + β_cos_2f · cos(4πt/τ)      (2nd harmonic)

Equivalent amplitude-phase form, used consistently across the pupil, saliva
and circadian cosinors of this package:

    y(t) = β₀ + A · cos(2πt/τ + φ)

so that φ is the phase angle and the peak occurs at t = -φ/2π · τ.

    amplitude    = √(α_sin²  + β_cos²)
    amplitude_2f = √(α_sin_2f² + β_cos_2f²)
    phase_rad    = atan2(-α_sin,  β_cos)         (radians, phase angle φ)
    phase_rad_2f = atan2(-α_sin_2f, β_cos_2f)        (radians, phase angle φ_2f)
    phase_hours    = estimated peak time from protocol start, i.e.
                     (-phase_rad / 2π · τ) mod τ            (hours)
    phase_hours_2f = (-phase_rad_2f / 2π · (τ/2)) mod (τ/2) (hours, period = τ/2)

The period τ is scanned across a range (default: 22–28 h). For each τ, the
parameter estimates and fit statistics are stored. The best τ per participant
is selected based on a chosen metric (default: AIC).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

from civibeplr.analysis_io import (
    DEFAULT_PUPIL_PATH,
    PUPIL_PARTICIPANT_COL,
    load_pupil_results,
)
from civibeplr.latex_table_creation import format_latex_variables
from civibeplr.utils_cosinor import circ_mean, circ_sd
from civibeplr.vis_io import _in_jupyter, apply_font_style

apply_font_style()


DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = DEFAULT_PROJECT_DIR / "data" / "raw" / "clean_saliva_data.csv"

DEFAULT_PARTICIPANT_COL = "participant_id"
DEFAULT_SESSION_COL = "session_id"
# The light-exposure (LE) and dim-light (DL) evenings were collected on other dates
# but share the FD standard_datetime axis, so pooling them would overlay three
# evenings. They are excluded, together with rows without a session label.
DEFAULT_EXCLUDED_SESSIONS = ("LE", "DL")
DEFAULT_OUTCOME_COL = "melatonin_concentration"
DEFAULT_TIME_COL = "standard_datetime"
DEFAULT_TIME_HOURS_COL = "hours_since_protocol_start"

DEFAULT_PERIOD_MIN = 22.0
DEFAULT_PERIOD_MAX = 28.0
DEFAULT_PERIOD_STEP = 1/60
DEFAULT_SELECTION_METRIC = "aic"
DEFAULT_SELECTION_METRICS = ["aic", "bic", "rmse", "rsquared", "adjrsquared"]

def load_clean_saliva_data(
    data_path: Path | None = None,
    excluded_sessions: tuple[str, ...] = DEFAULT_EXCLUDED_SESSIONS,
) -> pd.DataFrame:
    """Load clean saliva data from CSV.

    Melatonin concentrations are used exactly as reported by the laboratory.
    Values below the assay's detection limit (0.2 pg/mL) or lower limit of
    quantification (0.5 pg/mL; quantification range 0.5-50 pg/mL),
    including values reported as 0, are neither substituted nor excluded.

    Parameters
    ----------
    data_path : Path | None
        Path to the clean saliva data CSV. If None, uses the default path.
    excluded_sessions : tuple[str, ...]
        Session IDs to drop (default: LE and DL evenings). Rows with an empty
        session ID are always dropped.

    Returns
    -------
    DataFrame
        Loaded saliva data.
    """
    data_path = data_path or DEFAULT_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Clean saliva data not found at: {data_path}")

    df = pd.read_csv(data_path)
    # IDs < 200 are pilot/test sessions not part of the main study cohort
    if DEFAULT_PARTICIPANT_COL in df.columns:
        df = df[df[DEFAULT_PARTICIPANT_COL] >= 200]
    if DEFAULT_SESSION_COL not in df.columns:
        raise KeyError(f"Missing session column: {DEFAULT_SESSION_COL}")
    session = df[DEFAULT_SESSION_COL]
    df = df[session.notna() & ~session.isin(excluded_sessions)]
    print(f"Loaded clean saliva data: {len(df)} rows, {len(df.columns)} columns")

    # Drop unnamed index column if present
    if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
        df = df.drop(columns=[df.columns[0]])
    return df


def add_hours_since_protocol_start(
    df: pd.DataFrame,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    standard_datetime_col: str = DEFAULT_TIME_COL,
    time_hours_col: str = DEFAULT_TIME_HOURS_COL,
) -> pd.DataFrame:
    """Add hours since protocol start per participant based on a datetime column.

    Parameters
    ----------
    df : DataFrame
        Input saliva data.
    participant_col : str
        Column name identifying participants.
    standard_datetime_col : str
        Column name with standard datetime values.
    time_hours_col : str
        Name of the derived hours-since-start column.

    Returns
    -------
    DataFrame
        DataFrame with the hours-since-start column added.
    """
    if time_hours_col in df.columns:
        return df

    if standard_datetime_col not in df.columns:
        raise KeyError(f"Missing time column: {standard_datetime_col}")

    df = df.copy()
    df[standard_datetime_col] = pd.to_datetime(df[standard_datetime_col], errors="coerce")

    if df[standard_datetime_col].isna().all():
        raise ValueError(f"Could not parse any timestamps in column: {standard_datetime_col}")

    # Compute hours since protocol start per participant by finding the minimum timestamp per participant and calculating differences
    df[time_hours_col] = (
        df.groupby(participant_col, dropna=False)[standard_datetime_col]
        .transform(lambda series: (series - series.min()).dt.total_seconds() / 3600.0)
    )
    return df


def _nan_cosinor_result(tau: float) -> dict[str, float]:
    """Return a NaN-filled result dictionary for a failed fit.

    Parameters
    ----------
    tau : float
        Period (hours) used for the attempted fit.

    Returns
    -------
    dict[str, float]
        Result dictionary with NaN values for parameters and statistics.
    """
    return {
        "tau_hours": tau,
        "beta_intercept": np.nan,
        "alpha_sin": np.nan,
        "beta_cos": np.nan,
        "alpha_sin_2f": np.nan,
        "beta_cos_2f": np.nan,
        "amplitude": np.nan,
        "amplitude_2f": np.nan,
        "phase_rad": np.nan,
        "phase_rad_2f": np.nan,
        "phase_hours": np.nan,
        "phase_hours_2f": np.nan,
        "rsquared": np.nan,
        "adjrsquared": np.nan,
        "aic": np.nan,
        "bic": np.nan,
        "rmse": np.nan,
        "n_obs": np.nan,
    }


def fit_cosinor_for_tau(
    time_hours: np.ndarray,
    outcome: np.ndarray,
    tau: float,
) -> dict[str, float]:
    """Fit cosinor model for a fixed period τ (hours).

    Parameters
    ----------
    time_hours : array
        Time values in hours since protocol start.
    outcome : array
        Outcome values (e.g., hormone concentration).
    tau : float
        Period in hours.

    Returns
    -------
    dict[str, float]
        Parameter estimates (fundamental + 2nd harmonic) and fit statistics
        for the given τ.
    """
    # Drop rows where either time or outcome is NaN before fitting
    valid_mask = ~(np.isnan(time_hours) | np.isnan(outcome))
    time_clean = time_hours[valid_mask]
    outcome_clean = outcome[valid_mask]

    # 5 parameters + 2 degrees of freedom for residual variance estimation
    if len(outcome_clean) < 7:
        return _nan_cosinor_result(tau)

    # Angular frequency of the fundamental (rad/h)
    omega = 2.0 * np.pi / tau

    # Design matrix: intercept + fundamental (sin/cos) + 2nd harmonic (sin/cos)
    design_matrix = np.column_stack(
        [
            np.ones(len(time_clean)),
            np.sin(omega * time_clean),       # fundamental sin
            np.cos(omega * time_clean),       # fundamental cos
            np.sin(2.0 * omega * time_clean), # 2nd harmonic sin  (period = τ/2)
            np.cos(2.0 * omega * time_clean), # 2nd harmonic cos  (period = τ/2)
        ]
    )

    try:
        model = sm.OLS(outcome_clean, design_matrix)
        results = model.fit()
    except (np.linalg.LinAlgError, ValueError):
        return _nan_cosinor_result(tau)

    beta_intercept, alpha_sin, beta_cos, alpha_sin_2f, beta_cos_2f = results.params

    # Amplitude = distance from origin in the (α_sin, β_cos) phasor space
    amplitude = np.sqrt(alpha_sin**2 + beta_cos**2)
    amplitude_2f = np.sqrt(alpha_sin_2f**2 + beta_cos_2f**2)

    # Convention: y = β₀ + A·cos(2πt/τ + φ), so φ = atan2(-α_sin, β_cos)
    # and the peak occurs at t = -φ/ω.
    phase_rad = np.arctan2(-alpha_sin, beta_cos)
    phase_rad_2f = np.arctan2(-alpha_sin_2f, beta_cos_2f)

    # Estimated peak time, in hours from protocol start, wrapped into one
    # period so the value is forward-looking (always in [0, period)).
    # fundamental period = τ;  2nd harmonic period = τ/2
    phase_hours = np.mod(-phase_rad / (2.0 * np.pi) * tau, tau)
    phase_hours_2f = np.mod(-phase_rad_2f / (2.0 * np.pi) * (tau / 2.0), tau / 2.0)

    return {
        "tau_hours": tau,
        "beta_intercept": beta_intercept,
        "alpha_sin": alpha_sin,
        "beta_cos": beta_cos,
        "alpha_sin_2f": alpha_sin_2f,
        "beta_cos_2f": beta_cos_2f,
        "amplitude": amplitude,
        "amplitude_2f": amplitude_2f,
        "phase_rad": phase_rad,
        "phase_rad_2f": phase_rad_2f,
        "phase_hours": phase_hours,
        "phase_hours_2f": phase_hours_2f,
        "rsquared": results.rsquared,
        "adjrsquared": results.rsquared_adj,
        "aic": results.aic,
        "bic": results.bic,
        "rmse": float(np.sqrt(results.mse_resid)),
        "n_obs": len(outcome_clean),
    }


def _select_best_fit(
    fits_df: pd.DataFrame,
    metrics: list[str] | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Select the best τ per participant based on one or more metrics.

    Parameters
    ----------
    fits_df : DataFrame
        Fit results for all tested τ values.
    metrics : list[str] | str
        Metric name(s) used to select the best τ per participant; default is ["aic", "bic", "rmse", "rsquared", "adjrsquared"]

    Returns
    -------
    dict[str, DataFrame]
        Mapping from metric name to a best-fit DataFrame.
    """
    if metrics is None:
        metrics = DEFAULT_SELECTION_METRICS
    if isinstance(metrics, str):
        metrics = [metrics]

    metric_set = {metric.lower() for metric in metrics}
    valid_metrics = {"aic", "bic", "rmse", "rsquared", "adjrsquared"}
    if not metric_set.issubset(valid_metrics):
        raise ValueError(
            "selection_metrics must be a subset of: aic, bic, rmse, rsquared, adjrsquared"
        )

    best_by_metric: dict[str, pd.DataFrame] = {}
    for metric in metric_set:
        # Drop participants with no valid fit for this metric
        df = fits_df.dropna(subset=[metric])
        if df.empty:
            best_by_metric[metric] = df.copy()
            continue
        # AIC, BIC, RMSE: lower is better → idxmin
        # R², adj-R²: higher is better → idxmax
        if metric in {"aic", "bic", "rmse"}:
            idx = df.groupby(DEFAULT_PARTICIPANT_COL, dropna=False)[metric].idxmin()
        else:
            idx = df.groupby(DEFAULT_PARTICIPANT_COL, dropna=False)[metric].idxmax()
        best_by_metric[metric] = df.loc[idx].reset_index(drop=True)

    return best_by_metric


def plot_best_fit_per_participant(
    df: pd.DataFrame,
    best_fits: pd.DataFrame,
    outcome_col: str,
    time_hours_col: str,
    output_path: Path | None = DEFAULT_PROJECT_DIR / "results" / "figures" / "saliva_cosinor" / "fig_best_fits_per_participant.png",
    n_cols: int = 3,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot raw data with best-fit cosinor overlay in a single multi-panel figure.

    One subplot is created per participant. Panels are arranged in a grid with
    ``n_cols`` columns. The figure is saved as a single PNG file.

    Parameters
    ----------
    df : DataFrame
        Clean saliva data with one row per sample.
    best_fits : DataFrame
        Best-fit rows per participant (output of selection step).
    outcome_col : str
        Column name of the hormone concentration to plot.
    time_hours_col : str
        Column containing hours since protocol start.
    output_path : Path | None
        Full path (including filename) for saving the figure.
        If None, the plot is shown interactively instead of saved.
        Ignored when *ax* is provided.
    n_cols : int
        Number of columns in the subplot grid.
    ax : Axes | None
        If provided, draw on this single axes instead of creating a new
        figure. Only works when plotting a single participant.

    Returns
    -------
    (Figure, ndarray of Axes)
        The matplotlib figure and the 2-D array of axes.
    """
    # Collect only participants that have a matching best-fit row
    participant_ids = [
        pid
        for pid, _ in df.groupby(DEFAULT_PARTICIPANT_COL, dropna=False)
        if not best_fits[best_fits[DEFAULT_PARTICIPANT_COL] == pid].empty
    ]
    n_participants = len(participant_ids)
    if n_participants == 0:
        raise ValueError("No participants with valid best-fit rows found.")

    # Compute grid dimensions
    n_cols = min(n_cols, n_participants)
    n_rows = int(np.ceil(n_participants / n_cols))

    external_ax = ax is not None
    if external_ax:
        fig = ax.get_figure()
        axes = np.array([[ax]])
    else:
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(4.5 * n_cols, 3.5 * n_rows),
            squeeze=False, # return 2D array of axes for consistent indexing
            sharey=True,
        )

    for idx, participant_id in enumerate(participant_ids):
        ax = axes[idx // n_cols, idx % n_cols]
        group = df[df[DEFAULT_PARTICIPANT_COL] == participant_id]
        fit = best_fits[best_fits[DEFAULT_PARTICIPANT_COL] == participant_id].iloc[0]

        tau = fit["tau_hours"]
        beta_intercept = fit["beta_intercept"]
        alpha_sin = fit["alpha_sin"]
        beta_cos = fit["beta_cos"]
        alpha_sin_2f = fit["alpha_sin_2f"]
        beta_cos_2f = fit["beta_cos_2f"]

        time_hours = group[time_hours_col].values.astype(float)
        outcome = group[outcome_col].values.astype(float)

        # Reconstruct the fitted curve on a dense time grid using stored β coefficients
        time_grid = np.linspace(np.nanmin(time_hours), np.nanmax(time_hours), 400)
        omega = 2.0 * np.pi / tau
        fit_curve = (
            beta_intercept
            + alpha_sin * np.sin(omega * time_grid)           # fundamental
            + beta_cos * np.cos(omega * time_grid)
            + alpha_sin_2f * np.sin(2.0 * omega * time_grid)   # 2nd harmonic
            + beta_cos_2f * np.cos(2.0 * omega * time_grid)
        )

        ax.scatter(time_hours, outcome, s=14, alpha=0.7, label="Raw data")
        ax.plot(time_grid, fit_curve, color="black", lw=1.5, label=f"τ={tau:.2f} h")

        # Design
        ax.set_title(f"Participant {participant_id}")
        ax.set_xlabel("Hours since protocol start [h]") if idx // n_cols == n_rows - 1 else ax.set_xlabel("")
        y_label = (
            "Salivary melatonin\nconcentration [pg/mL]"
            if outcome_col == "melatonin_concentration"
            else "Salivary cortisol\nconcentration [ng/mL]"
        )
        ax.set_ylabel(y_label) if idx % n_cols == 0 else ax.set_ylabel("")
        ax.legend(frameon=False)
        ax.set_xticks(np.arange(0, np.ceil(np.nanmax(time_hours)) + 1, 5))
        ax.grid(axis="y", ls="-", lw=0.5, alpha=0.5)
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)

    for idx in range(n_participants, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    if not external_ax:
        fig.tight_layout()
        if output_path is not None:
            output_path.parent.mkdir(parents=False, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"    Saved multi-panel figure in: {output_path}")
            if not _in_jupyter():
                plt.close(fig)
        else:
            plt.show()

    return fig, axes


def plot_metric_vs_tau(
    fits_df: pd.DataFrame,
    metric: str = "rsquared",
    output_path: Path | None = DEFAULT_PROJECT_DIR / "results" / "figures" / "saliva_cosinor" / "fig_metric_vs_tau.png",
    n_cols: int = 3,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a fit metric against τ for each participant in a multi-panel figure.

    Each subplot shows the metric curve over the scanned τ range for one
    participant. A red vertical line marks τ = 0 on the x-axis (reference).
    A black dashed vertical line marks the τ that maximises / minimises the
    metric, annotated with the numeric value of that best τ.

    Parameters
    ----------
    fits_df : DataFrame
        Long-format DataFrame of all (participant × τ) fit results, as
        returned by the period-scanning loop in ``main()``.
    metric : str
        Column name of the metric to plot on the y-axis (e.g. "rsquared",
        "aic", "bic", "adjrsquared", "rmse").
    output_path : Path | None
        Full path (including filename) for saving the figure.
        If None, the plot is shown interactively instead of saved.
        Ignored when *ax* is provided.
    n_cols : int
        Number of columns in the subplot grid.
    ax : Axes | None
        If provided, draw on this single axes instead of creating a new
        figure. Only works when plotting a single participant.

    Returns
    -------
    (Figure, ndarray of Axes)
        The matplotlib figure and the 2-D array of axes.
    """
    if metric not in fits_df.columns:
        raise ValueError(f"Metric '{metric}' not found in fits_df columns: {list(fits_df.columns)}")

    # Metrics where lower = better (best τ is the minimum)
    lower_is_better = {"aic", "bic", "rmse"}

    participant_ids = sorted(fits_df[DEFAULT_PARTICIPANT_COL].dropna().unique())
    n_participants = len(participant_ids)
    if n_participants == 0:
        raise ValueError("fits_df contains no participant data.")

    # Compute grid dimensions
    n_cols = min(n_cols, n_participants)
    n_rows = int(np.ceil(n_participants / n_cols))

    external_ax = ax is not None
    if external_ax:
        fig = ax.get_figure()
        axes = np.array([[ax]])
    else:
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(3.0 * n_cols, 2.5 * n_rows),
            squeeze=False,
            sharey=True, 
        )

    for idx, participant_id in enumerate(participant_ids):
        ax = axes[idx // n_cols, idx % n_cols]
        pdata = fits_df[fits_df[DEFAULT_PARTICIPANT_COL] == participant_id].dropna(subset=[metric])

        if pdata.empty:
            ax.set_visible(False)
            continue

        tau_vals = pdata["tau_hours"].values
        metric_vals = pdata[metric].values

        # Find the best τ (min for error metrics, max for goodness-of-fit)
        if metric.lower() in lower_is_better:
            best_idx = np.argmin(metric_vals)
        else:
            best_idx = np.argmax(metric_vals)
        best_tau = tau_vals[best_idx]
        best_val = metric_vals[best_idx]

        # Plot metric curve
        ax.plot(tau_vals, metric_vals, lw=1.4, color="steelblue")

        # Red vertical line at τ = 0 (reference / x-axis zero)
        # ax.axhline(0, color="red", lw=1.0, ls="-", alpha=0.7, label="τ = 0")

        # Dashed black vertical line at the best τ
        ax.axvline(best_tau, color="black", lw=1.2, ls="--")

        # Annotate best τ value near the top of the panel
        ax.annotate(
            f"τ* = {best_tau:.2f} h",
            xy=(best_tau, best_val),
            xytext=(0.97, 0.95),
            textcoords="axes fraction",
            ha="right", va="top",
            #fontsize=7,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        )

        ax.set_title(f"Participant {participant_id}")
        ax.set_xlabel("τ [h]")
        ax.set_ylabel("R²" if metric == "rsquared" else metric)
        ax.set_xticks(np.arange(DEFAULT_PERIOD_MIN, DEFAULT_PERIOD_MAX + 1e-6, 1))
        ax.grid(axis="y", ls="-", lw=0.5, alpha=0.5)
        # ax.spines["top"].set_visible(False)
        # ax.spines["right"].set_visible(False)

    for idx in range(n_participants, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    if not external_ax:
        fig.tight_layout()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"    Saved metric-vs-tau figure in:   {output_path}")
            if not _in_jupyter():
                plt.close(fig)
        else:
            plt.show()

    return fig, axes


def plot_saliva_cosinor_summary(
    df: pd.DataFrame,
    fits_df: pd.DataFrame,
    best_df: pd.DataFrame,
    exemplar_participant,
    outcome_col: str,
    time_hours_col: str,
    metric: str = "rsquared",
    output_path: Path | None = DEFAULT_PROJECT_DIR / "results" / "figures" / "saliva_cosinor" / "fig_saliva_cosinor_summary.png",
) -> tuple[plt.Figure, np.ndarray]:
    """Create a 3×1 summary figure: best fit, metric vs τ, τ histogram."""
    fig, axes = plt.subplots(3, 1, figsize=(4.5, 10.5))

    plot_best_fit_per_participant(
        df[df[DEFAULT_PARTICIPANT_COL] == exemplar_participant],
        best_df[best_df[DEFAULT_PARTICIPANT_COL] == exemplar_participant],
        outcome_col=outcome_col,
        time_hours_col=time_hours_col,
        n_cols=1,
        ax=axes[0],
    )
    plot_metric_vs_tau(
        fits_df[fits_df[DEFAULT_PARTICIPANT_COL] == exemplar_participant],
        metric=metric,
        n_cols=1,
        ax=axes[1],
    )
    conc_values = df[outcome_col].dropna().values
    conc_mean = float(np.mean(conc_values))
    conc_std = float(np.std(conc_values, ddof=1))
    axes[2].hist(conc_values, bins="auto", edgecolor="white", color="steelblue", alpha=0.5)
    axes[2].axvline(conc_mean, color="crimson", lw=1.8, ls="--",
                    label=f"Mean = {conc_mean:.1f}, SD = {conc_std:.1f}")
    y_label = (
        "Salivary melatonin\nconcentration [pg/mL]"
        if outcome_col == "melatonin_concentration"
        else "Salivary cortisol\nconcentration [ng/mL]"
    )
    axes[2].set_xlabel(y_label)
    axes[2].set_ylabel("Count")
    axes[2].set_title("Overall study sample")
    axes[2].legend(frameon=False)

    # Add panel labels (A, B, C) in the upper-left corner of each subplot
    for label, ax in zip("ABC", axes):
        ax.text(
            -0.1, 1.05, label,
            transform=ax.transAxes,
            fontsize=14, fontweight="bold",
            va="bottom", ha="right",
        )

    fig.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=False, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"    Saved summary figure in: {output_path}")
        if not _in_jupyter():
            plt.close(fig)
    else:
        plt.show()

    return fig, axes


def plot_tau_histogram(
    best_df: pd.DataFrame,
    output_path: Path | None = DEFAULT_PROJECT_DIR / "results" / "figures" / "saliva_cosinor" / "fig_tau_histogram.png",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of best-fit τ values with the mean marked.

    Parameters
    ----------
    best_df : DataFrame
        Best-fit results with a ``tau_hours`` column (one row per participant).
    output_path : Path | None
        Full path (including filename) for saving the figure.
        If None, the plot is shown interactively.
        Ignored when *ax* is provided.
    ax : Axes | None
        If provided, draw on this axes instead of creating a new figure.

    Returns
    -------
    (Figure, Axes)
    """
    tau_values = best_df["tau_hours"].dropna().values
    tau_mean = float(np.mean(tau_values))

    external_ax = ax is not None
    if not external_ax:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
    else:
        fig = ax.get_figure()

    ax.hist(tau_values, bins="auto", edgecolor="white", color="steelblue", alpha=0.5)
    ax.axvline(tau_mean, color="crimson", lw=1.8, ls="--",
               label=f"Mean = {tau_mean:.2f} h")
    ax.set_xlabel("τ [h]")
    ax.set_ylabel("Count")
    ax.set_title("Overall study sample")
    ax.legend()

    if not external_ax:
        fig.tight_layout()
        if output_path is not None:
            output_path.parent.mkdir(parents=False, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"    Saved τ histogram in: {output_path}")
            if not _in_jupyter():
                plt.close(fig)
        else:
            plt.show()

    return fig, ax


# Columns included in the LaTeX summary table (τ + fit statistics; no model params)
_LATEX_STAT_COLS = [
    "participant_id",
    "tau_hours",
    "amplitude",
    "phase_rad",
    "phase_hours",
    "amplitude_2f",
    "phase_rad_2f",
    "phase_hours_2f",
    "rsquared",
    "adjrsquared",
    "aic",
    "bic",
    "rmse",
    "n_obs",
    "melatonin_mean",
    "melatonin_sd",
    "melatonin_median",
    "melatonin_min",
    "melatonin_max",
]

# Human-readable column headers for the LaTeX table
_LATEX_COL_LABELS = {
    "participant_id": "Participant",
    "tau_hours": r"$\hat{\tau}$ [h]",
    "amplitude": r"$A_{f}$ [pg/mL]",
    "phase_rad": r"$\phi_{f}$ [rad]",
    "phase_hours": r"$t_{\mathrm{peak}}$ [h]",
    "amplitude_2f": r"$A_{2f}$ [pg/mL]",
    "phase_rad_2f": r"$\phi_{2f}$ [rad]",
    "phase_hours_2f": r"$t_{\mathrm{peak},2f}$ [h]",
    "rsquared": r"$R^2$",
    "adjrsquared": r"$R^2_{\mathrm{adj}}$",
    "aic": "AIC",
    "bic": "BIC",
    "rmse": "RMSE",
    "n_obs": "$n$",
    "melatonin_mean": "Mean",
    "melatonin_sd": "SD",
    "melatonin_median": "Median",
    "melatonin_min": "Min",
    "melatonin_max": "Max",
}

# Phase angles straddle +/-pi (notably the second harmonic), so their mean and SD
# across participants use circular statistics.
_PHASE_COLS = ("phase_rad", "phase_rad_2f")

# Number of decimal places per column
_LATEX_COL_DECIMALS = {
    "tau_hours": 2,
    "amplitude": 3,
    "phase_rad": 3,
    "phase_hours": 2,
    "amplitude_2f": 3,
    "phase_rad_2f": 3,
    "phase_hours_2f": 2,
    "rsquared": 3,
    "adjrsquared": 3,
    "aic": 1,
    "bic": 1,
    "rmse": 3,
    "n_obs": 0,
    "melatonin_mean": 2,
    "melatonin_sd": 2,
    "melatonin_median": 2,
    "melatonin_min": 2,
    "melatonin_max": 2,
}


def save_latex_table(
    best_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write a landscape LaTeX table of τ, amplitude/phase, and fit statistics.

    The table uses a two-level header: a top group row with \\multicolumn
    spans for "Circadian rhythm parameters" and "Goodness-of-fit statistics",
    each underlined with \\cmidrule, followed by the individual column labels.
    The table is wrapped in ``sidewaystable`` (requires the ``rotating``
    package) so it fits on a portrait page when rotated.

    Columns included: participant, τ̂, amplitude and phase (fundamental and 2nd
    harmonic), R², adj-R², AIC, BIC, RMSE, n.

    Parameters
    ----------
    best_df : DataFrame
        Best-fit rows per participant (primary AIC selection).
    output_path : Path
        Destination ``.tex`` file.
    """
    # Keep only columns that are present and relevant
    cols = [c for c in _LATEX_STAT_COLS if c in best_df.columns]
    table = best_df[cols].copy().sort_values("participant_id")

    # Format numeric columns
    for col, decimals in _LATEX_COL_DECIMALS.items():
        if col not in table.columns:
            continue
        if decimals == 0:
            table[col] = table[col].apply(
                lambda v: str(int(v)) if pd.notna(v) else "--"
            )
        else:
            fmt = f"{{:.{decimals}f}}"
            table[col] = table[col].apply(
                lambda v, f=fmt: f.format(v) if pd.notna(v) else "--"
            )

    # Build column spec: l for participant, r for all numerics
    col_spec = "c" + "r" * (len(cols) - 1)

    # -----------------------------------------------------------------
    # Two-level header: group spans + individual column labels
    # -----------------------------------------------------------------
    # Define which columns belong to each group (in display order)
    _param_group = [
        "tau_hours", "amplitude", "phase_rad", "phase_hours",
        "amplitude_2f", "phase_rad_2f", "phase_hours_2f",
    ]
    _fit_group = ["rsquared", "adjrsquared", "aic", "bic", "rmse", "n_obs"]
    _mel_group = ["melatonin_mean", "melatonin_sd", "melatonin_median", "melatonin_min", "melatonin_max"]

    param_cols_present = [c for c in _param_group if c in cols]
    fit_cols_present   = [c for c in _fit_group   if c in cols]
    mel_cols_present   = [c for c in _mel_group   if c in cols]

    # 1-based column indices for \cmidrule
    def _col_range(group: list[str]) -> tuple[int, int]:
        start = cols.index(group[0]) + 1
        end   = cols.index(group[-1]) + 1
        return start, end

    param_start, param_end = _col_range(param_cols_present)
    fit_start,   fit_end   = _col_range(fit_cols_present)

    n_param = len(param_cols_present)
    n_fit   = len(fit_cols_present)

    # Group header row: empty cell for participant column, then three spans
    group_header = (
        f"    & \\multicolumn{{{n_param}}}{{c}}{{Circadian rhythm parameters}}"
        f" & \\multicolumn{{{n_fit}}}{{c}}{{Goodness-of-fit statistics}}"
    )
    # Cmidrules underneath each group header (trim left/right for spacing)
    cmidrules = (
        f"    \\cmidrule(lr){{{param_start}-{param_end}}}"
        f" \\cmidrule(lr){{{fit_start}-{fit_end}}}"
    )
    if mel_cols_present:
        mel_start, mel_end = _col_range(mel_cols_present)
        n_mel = len(mel_cols_present)
        group_header += f" & \\multicolumn{{{n_mel}}}{{c}}{{Melatonin concentration [pg/mL]}}"
        cmidrules += f" \\cmidrule(lr){{{mel_start}-{mel_end}}}"
    group_header += " \\\\"

    # Individual column label row
    col_header = " & ".join(_LATEX_COL_LABELS.get(c, c) for c in cols)

    lines = [
        r"% Requires: \usepackage{booktabs, rotating} in preamble",
        r"\begin{sidewaystable}",
        r"  \centering",
        # \small with narrower column gaps keeps all 19 columns on the rotated page
        # while staying legible at normal reading zoom.
        r"  \small",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \caption{Best-fit period, circadian rhythm parameters, and goodness-of-fit"
        r" statistics per participant. $\hat{\tau}$ is the period selected by AIC from"
        r" the 22--28\,h scan; $A$ and $\phi$ are the amplitude and phase of the"
        r" fundamental ($f$) and second harmonic ($2f$) of the bi-harmonic cosinor,"
        r" and $t_{\mathrm{peak}}$ is the corresponding peak time in hours from"
        r" protocol start. $n$ is the number of saliva samples, and the concentration"
        r" columns summarise the raw values entering the fit. The final rows give the"
        r" mean and SD across participants (circular mean and circular SD for $\phi$).}",
        r"  \label{tab:tab_saliva_cosinor_fit}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        group_header,
        cmidrules,
        f"    {col_header} \\\\",
        r"    \midrule",
    ]

    for _, row in table.iterrows():
        data_row = " & ".join(str(row[c]) for c in cols)
        lines.append(f"    {data_row} \\\\")

    # M (SD) summary row using the original numeric DataFrame (before formatting)
    numeric_cols = [c for c in cols if c != "participant_id"]
    mean_vals = best_df[numeric_cols].mean()
    sd_vals = best_df[numeric_cols].std()
    for col in _PHASE_COLS:
        if col in numeric_cols:
            mean_vals[col] = circ_mean(best_df[col])
            sd_vals[col] = circ_sd(best_df[col])

    mean_cells = ["$M$"]
    sd_cells = ["($SD$)"]
    for col in numeric_cols:
        decimals = _LATEX_COL_DECIMALS.get(col, 2)
        mean_cells.append(f"{mean_vals[col]:.{decimals}f}")
        sd_cells.append(f"({sd_vals[col]:.{decimals}f})")

    lines.append(r"    \midrule")
    lines.append(f"    {' & '.join(mean_cells)} \\\\")
    lines.append(f"    {' & '.join(sd_cells)} \\\\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{sidewaystable}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    Saved LaTeX table in:   {output_path}")


def main(
    data_path: Path | None = None,
    outcome_col: str = DEFAULT_OUTCOME_COL,
    time_col: str = DEFAULT_TIME_COL,
    time_hours_col: str = DEFAULT_TIME_HOURS_COL,
    period_min: float = DEFAULT_PERIOD_MIN,
    period_max: float = DEFAULT_PERIOD_MAX,
    period_step: float = DEFAULT_PERIOD_STEP,
    selection_metrics: list[str] | str | None = None,
    output_dir: Path | None = DEFAULT_PROJECT_DIR / "results",
    plot: bool = True,
    pupil_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run cosinor analysis across a range of periods for saliva data.

    Parameters
    ----------
    data_path : Path | None
        Path to the clean saliva data CSV.
    outcome_col : str
        Column name of the outcome to fit.
    time_col : str
        Datetime column for computing hours since protocol start.
    time_hours_col : str
        Optional precomputed hours-since-start column.
    period_min : float
        Minimum τ (hours) to test.
    period_max : float
        Maximum τ (hours) to test.
    period_step : float
        Step size for τ (hours).
    selection_metric : list[str] | str
        Metric(s) for selecting the best τ per participant.
    output_dir : Path | None
        Output directory for tables and figures.
    plot : bool
        Whether to generate plots.

    Returns
    -------
    (DataFrame, DataFrame)
        All fit rows and the best-fit rows for the primary metric.
    """
    # 1. Load and validate data
    df = load_clean_saliva_data(data_path)

    if DEFAULT_PARTICIPANT_COL not in df.columns:
        raise KeyError(f"Missing participant column: {DEFAULT_PARTICIPANT_COL}")
    if outcome_col not in df.columns:
        raise KeyError(f"Missing outcome column: {outcome_col}")

    # 2. Add hours-since-protocol-start if not already present
    # The cosinor model needs a continuous time axis anchored at t=0 for
    # each participant (first sample = 0 h).
    df = add_hours_since_protocol_start(
        df,
        participant_col=DEFAULT_PARTICIPANT_COL,
        standard_datetime_col=time_col,
        time_hours_col=time_hours_col,
    )

    # 3. Scan periods and fit the cosinor model for each (participant, τ)
    # +1e-6 ensures period_max is included despite floating-point truncation
    periods = np.arange(period_min, period_max + 1e-6, period_step)

    all_rows: list[dict[str, float]] = []
    for participant_id, group in df.groupby(DEFAULT_PARTICIPANT_COL, dropna=False):
        time_hours = group[time_hours_col].values.astype(float)
        outcome = group[outcome_col].values.astype(float)
        # Fit the two-harmonic cosinor model at every candidate period
        for tau in periods:
            fit = fit_cosinor_for_tau(time_hours, outcome, tau)
            fit[DEFAULT_PARTICIPANT_COL] = participant_id
            all_rows.append(fit)

    # Collect all (participant × τ) fit results into one long-format DataFrame
    fits_df = pd.DataFrame(all_rows)

    # 4. Select best τ per participant for each evaluation metric
    # Apply default metric list if none was supplied
    if selection_metrics is None:
        selection_metrics = DEFAULT_SELECTION_METRICS
    # Returns a dict mapping each metric name to a per-participant best-fit DataFrame
    best_by_metric = _select_best_fit(fits_df, metrics=selection_metrics)
    best_df = best_by_metric.get("aic", pd.DataFrame())

    # 5. Filter to participants that also have pupil data
    pupil_path = pupil_path or DEFAULT_PUPIL_PATH
    if pupil_path.exists():
        pupil_ids = set(load_pupil_results(pupil_path)[PUPIL_PARTICIPANT_COL].unique())
        saliva_ids_before = set(df[DEFAULT_PARTICIPANT_COL].unique())
        dropped = saliva_ids_before - pupil_ids
        if dropped:
            print(
                f"  Saliva-only participants (excluded — no valid pupillometry): "
                f"{sorted(dropped)}"
            )
        df       = df      [df      [DEFAULT_PARTICIPANT_COL].isin(pupil_ids)]
        fits_df  = fits_df [fits_df [DEFAULT_PARTICIPANT_COL].isin(pupil_ids)]
        best_df  = best_df [best_df [DEFAULT_PARTICIPANT_COL].isin(pupil_ids)]
        for metric, bdf in best_by_metric.items():
            best_by_metric[metric] = bdf[bdf[DEFAULT_PARTICIPANT_COL].isin(pupil_ids)]
    else:
        print(
            f"  Pupil cosinor results not found at {pupil_path} — "
            "skipping saliva-participant filter."
        )

    # 5b. Attach per-participant melatonin descriptive statistics to best_df
    conc_stats = (
        df.groupby(DEFAULT_PARTICIPANT_COL)[outcome_col]
        .agg(
            melatonin_mean="mean",
            melatonin_sd="std",
            melatonin_median="median",
            melatonin_min="min",
            melatonin_max="max",
        )
        .reset_index()
    )
    best_df = best_df.merge(conc_stats, on=DEFAULT_PARTICIPANT_COL, how="left")

    # 6. Calculate summary statistics across participants for the best-fit τ values
    if not best_df.empty:
        summary_stats = best_df.describe()
        for col in _PHASE_COLS:
            summary_stats.loc["mean", col] = circ_mean(best_df[col])
            summary_stats.loc["std", col] = circ_sd(best_df[col])
        print("\nSummary of best-fit τ values across participants:")
        print(summary_stats)

    # 7. Save results
    table_dir = output_dir / "tables" / "saliva_cosinor"
    table_dir.mkdir(parents=True, exist_ok=True)

    if not best_df.empty:
        # Save primary best-fit table (τ selected by AIC) — used for plotting
        best_fits_path = table_dir / "saliva_cosinor_best_fit.csv"
        best_df.to_csv(best_fits_path, index=False)
        print(f"    Saved primary best fits (AIC) in:   {best_fits_path}")

        # Save LaTeX summary table (τ + fit statistics, no model parameters)
        latex_path = table_dir / "saliva_cosinor_best_fit.tex"
        latex_path.parent.mkdir(parents=True, exist_ok=True)
        save_latex_table(best_df=best_df, output_path=latex_path)

    if not summary_stats.empty:
        # Save LaTeX \def variables (one per cell, for manuscript inline use)
        vars_path = output_dir / "tex" / "saliva_cosinor" / "vars_saliva_cosinor_summary.tex"
        vars_path.parent.mkdir(parents=True, exist_ok=True)
        summary_stats_reset = summary_stats.reset_index().rename(columns={"index": "statistic"})
        # The describe()-based Saliva<Stat>Melatonin* macros are statistics of per-participant
        # values (e.g. the mean of participant medians); the pooled macros describe all
        # raw concentrations entering the fits, i.e. the distribution in Figure 2C.
        conc = df[outcome_col].dropna()
        pooled = (
            f"\\def\\SalivaPooledMelatoninN{{{len(conc)}}}\n"
            f"\\def\\SalivaPooledMelatoninMean{{{conc.mean():.2f}}}\n"
            f"\\def\\SalivaPooledMelatoninMedian{{{conc.median():.2f}}}\n"
        )
        vars_path.write_text(
            format_latex_variables(summary_stats_reset, row_key_col="statistic", prefix="Saliva")
            + "\n" + pooled,
            encoding="utf-8",
        )
        print(f"    Saved LaTeX variables in: {vars_path}")
    else:
        print("No valid best fits found for the primary metric (AIC).")

    # ------------------------------------------------------------------ #
    # 6. Plot best-fit curves and metric-vs-τ panels (optional)           #
    # ------------------------------------------------------------------ #
    if plot:
        fig_dir = output_dir / "figures" / "saliva_cosinor"
        # Plot 1: multi-panel raw data with best-fit cosinor overlay
        if not best_df.empty:
            plot_best_fit_per_participant(
                df,
                best_df,
                outcome_col=outcome_col,
                time_hours_col=time_hours_col,
                output_path=fig_dir / "fig_participants_best_fit.png",
            )
            # Plot 1a: exemplar single-participant best fit (if any valid fits exist)
            # median row by position -- not a statistical median, just a representative middle case
            exemplar_participant = best_df[DEFAULT_PARTICIPANT_COL].iloc[len(best_df) // 2]
            # plot_best_fit_per_participant(
            #     df[df[DEFAULT_PARTICIPANT_COL] == exemplar_participant],
            #     best_df[best_df[DEFAULT_PARTICIPANT_COL] == exemplar_participant],
            #     outcome_col=outcome_col,
            #     time_hours_col=time_hours_col,
            #     output_path=fig_dir / f"fig_participant_{exemplar_participant}_best_fit.png",
            #     n_cols=1,
            # )

        # Plot 2: metric (R²) vs τ diagnostic, one subplot per participant
        plot_metric_vs_tau(
            fits_df,
            metric="rsquared",
            output_path=fig_dir / "fig_metric_vs_tau.png",
        )
        # Plot 2a: exemplar single-participant metric vs τ
        # if not best_df.empty:
        #     plot_metric_vs_tau(
        #         fits_df[fits_df[DEFAULT_PARTICIPANT_COL] == exemplar_participant],
        #         metric="rsquared",
        #         output_path=fig_dir / f"fig_participant_{exemplar_participant}_metric_vs_tau.png",
        #         n_cols=1,
        #     )

        # Plot 4: combined 1×3 summary (best fit, R² vs τ, τ histogram)
        if not best_df.empty:
            plot_saliva_cosinor_summary(
                df,
                fits_df,
                best_df,
                exemplar_participant=exemplar_participant,
                outcome_col=outcome_col,
                time_hours_col=time_hours_col,
                output_path=fig_dir / "fig_saliva_cosinor_summary.png",
            )

    return fits_df, best_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit cosinor models to saliva hormone concentrations.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help=f"Path to clean saliva data CSV (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--outcome-col",
        type=str,
        default=DEFAULT_OUTCOME_COL,
        help=f"Outcome column (default: {DEFAULT_OUTCOME_COL}).",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default=DEFAULT_TIME_COL,
        help=f"Datetime column used to compute hours since protocol start (default: {DEFAULT_TIME_COL}).",
    )
    parser.add_argument(
        "--time-hours-col",
        type=str,
        default=DEFAULT_TIME_HOURS_COL,
        help=f"Optional precomputed hours-since-protocol-start column (default: {DEFAULT_TIME_HOURS_COL}).",
    )
    parser.add_argument(
        "--period-min",
        type=float,
        default=DEFAULT_PERIOD_MIN,
        help=f"Minimum period to test in hours (default: {DEFAULT_PERIOD_MIN}).",
    )
    parser.add_argument(
        "--period-max",
        type=float,
        default=DEFAULT_PERIOD_MAX,
        help=f"Maximum period to test in hours (default: {DEFAULT_PERIOD_MAX}).",
    )
    parser.add_argument(
        "--period-step",
        type=float,
        default=DEFAULT_PERIOD_STEP,
        help=f"Period step in hours (default: {DEFAULT_PERIOD_STEP}).",
    )
    parser.add_argument(
        "--selection-metrics",
        type=str,
        default=DEFAULT_SELECTION_METRIC,
        help=(
            "Comma-separated metrics used to select best period per participant. "
            "Choices: aic, bic, rmse, rsquared, adjrsquared."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR / "results",
        help="Output directory for tables and figures.",
    )
    parser.add_argument(
        "--pupil-path",
        type=Path,
        default=None,
        help=(
            f"Path to pupil cosinor results CSV used to filter saliva participants "
            f"(default: {DEFAULT_PUPIL_PATH}). Pass an empty string to skip filtering."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation.",
    )

    args = parser.parse_args()

    selection_metrics = [
        metric.strip().lower()
        for metric in args.selection_metrics.split(",")
        if metric.strip()
    ]

    main(
        data_path=args.data_path,
        outcome_col=args.outcome_col,
        time_col=args.time_col,
        time_hours_col=args.time_hours_col,
        period_min=args.period_min,
        period_max=args.period_max,
        period_step=args.period_step,
        selection_metrics=selection_metrics,
        output_dir=args.output_dir,
        plot=not args.no_plot,
        pupil_path=args.pupil_path,
    )
