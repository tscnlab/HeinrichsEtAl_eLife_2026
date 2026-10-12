"""
Visualisation for the two-step cosinor analysis results.

Loads the results CSV produced by ``fit_pupil_trace_2step_cosinor_model``
and generates all diagnostic figures. Also loads the aggregated input data
for decomposition plots that overlay the model fit on raw traces.

Decomposition plots require settings (e.g., cutoff, filter_order, freq, etc.)
to be loaded exclusively from a saved JSON metadata file (see --load-settings-json).
Manual CLI arguments for these settings are ignored if a JSON file is provided.

Usage
-----
    python -m civibeplr.visualise_pupil_cosinor [options]

Options
-------
    --results-csv PATH      Results CSV (default: results/tables/pupil_cosinor/pupil_cosinor_2step_param.csv)
    --data PATH             Aggregated data CSV (default: data/aggregated/data.csv)
    --output-dir PATH       Output root; figures go to <output-dir>/figures/pupil_cosinor/
    --load-settings-json    Path to JSON file with decomposition settings (required for decomposition plots)
    --save-settings-json    Path to save current decomposition settings as JSON
    --plot-individual       Generate per-participant decomposition plots (slow)
    # The following are only used for parameter plots or for saving settings, not for decomposition if --load-settings-json is given:
    --cutoff HZ             Low-pass cutoff used during fitting (default: 0.2)
    --filter-order N        Butterworth filter order (default: 2)
    --freq HZ               Stimulus frequency (default: 0.4)
    --add-poststim-sec S    Post-stimulus buffer appended to fit window (default: 0; controlled by DEFAULT_ADD_POSTSTIM_SEC in analysis_io.py)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


from civibeplr.analysis_io import DEFAULT_ADD_POSTSTIM_SEC, DEFAULT_T_STIM_END, DEFAULT_T_STIM_START
from civibeplr.data_io import (
    DEFAULT_BLOCK_COL,
    DEFAULT_CONDITION_COL,
    DEFAULT_PARTICIPANT_COL,
    DEFAULT_TIME_COL,
    load_aggregated_data,
)
from civibeplr.fit_pupil_trace_2step_cosinor_model import (
    _CIRCULAR_COLS,
    DEFAULT_CUTOFF_HZ,
    DEFAULT_FILTER_ORDER,
    DEFAULT_FREQUENCY,
    DEFAULT_OUTCOME_COL,
    DEFAULT_PROJECT_DIR,
    DEFAULT_RESULTS_DIR,
    extract_tonic_and_residual,
    fit_cosinor_residual,
)
from civibeplr.utils_cosinor import _circ_mean, _circ_sem
from civibeplr.vis_io import apply_font_style, TRIAL_TYPE_COLORS, _in_jupyter, save_figure

apply_font_style()

DEFAULT_RESULTS_CSV = DEFAULT_PROJECT_DIR / "results" / "tables" / "pupil_cosinor" / "pupil_cosinor_2step_param.csv"
DEFAULT_SETTINGS_JSON = DEFAULT_RESULTS_DIR / "metadata" / "pupil_cosinor_2step_settings.json"

# Participant shown in the exemplary decomposition and parameter figures. Fixed by
# ID (not list position) so both figures show the same person and the file names
# match the manuscript captions.
EXEMPLARY_PARTICIPANT = 202

# ── Helper functions ─────────────────────────────────────────────────────────

def _compute_block_stats(
    cond_data: pd.DataFrame,
    col: str,
    block_col: str,
    is_circular: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (block_stats_df, mean_vals, sem_vals) for one condition/column."""
    if is_circular:
        block_stats = (
            cond_data.groupby(block_col)[col]
            .agg(circ_mean=_circ_mean, circ_sem=_circ_sem)
            .reset_index()
        )
        return block_stats, block_stats["circ_mean"], block_stats["circ_sem"]
    else:
        block_stats = (
            cond_data.groupby(block_col)[col].agg(["mean", "sem"]).reset_index()
        )
        return block_stats, block_stats["mean"], block_stats["sem"]


def _plot_block_traces_with_mean_sem(
    ax: plt.Axes,
    cond_data: pd.DataFrame,
    col: str,
    block_col: str,
    participant_col: str,
    color: str,
    block_ticks: list,
    is_circular: bool,
    plot_individual_traces: bool = True,
    label: str | None = None,
) -> None:
    """Draw individual participant traces and mean ± SEM across blocks on ax."""
    if plot_individual_traces:
        for pid in cond_data[participant_col].unique():
            p_data = cond_data[cond_data[participant_col] == pid].sort_values(block_col)
            ax.plot(p_data[block_col], p_data[col], color=color, lw=0.5)

    block_stats, mean_vals, sem_vals = _compute_block_stats(
        cond_data, col, block_col, is_circular
    )
    ax.plot(block_stats[block_col], mean_vals, color=color, marker="o", ms=4, lw=1.5, label=label)
    ax.fill_between(
        block_stats[block_col],
        mean_vals - sem_vals,
        mean_vals + sem_vals,
        color=color,
        alpha=0.2,
        edgecolor="none",
    )
    ax.set_xticks(block_ticks)
    ax.grid(True, axis="y", ls="-", lw=0.3, alpha=0.5)
    #ax.autoscale(enable=True, axis="y", tight=False)
    #ax.margins(y=0.1)


def plot_raincloud(
    results_df: pd.DataFrame,
    parameter: str,
    parameter_label: str | None = None,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    ax: plt.Axes | None = None,
    output_path: Path | None = None,
) -> None:
    """Raincloud plot for one parameter across all conditions.

    Each condition shows jittered per-participant means (left), a box-plot
    summary (centre, right half overlapping the KDE), and a half-violin
    (KDE, right).  Values are averaged across blocks before plotting.

    Parameters
    ----------
    ax : Axes, optional
        If provided, draw into this axes and skip figure creation / saving.
        Use this when embedding the plot inside a grid (see
        ``plot_raincloud_grid``).
    output_path : Path, optional
        Save destination when ``ax`` is None (standalone figure).
    """
    conditions = sorted(results_df[condition_col].dropna().unique())

    rng = np.random.default_rng(42)
    ylabel = parameter_label or parameter

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(2.0 * len(conditions), 4))

    for i, cond in enumerate(conditions):
        color = TRIAL_TYPE_COLORS.get(cond, "tab:blue")
        cond_df = results_df[results_df[condition_col] == cond]
        vals = (
            cond_df.groupby(participant_col, dropna=False)[parameter]
            .mean()
            .dropna()
            .values
        )
        if len(vals) < 2:
            continue

        # ── Jittered individual points (left) ─────────────────────────────
        jitter = rng.uniform(-0.04, 0.04, size=len(vals))
        ax.scatter(
            i - 0.22 + jitter,
            vals,
            color=color,
            s=22,
            alpha=0.6,
            zorder=3,
            linewidths=0,
        )

        # ── Box-plot elements (centre, right half overlaps KDE) ────────────
        q1, median, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        whisker_lo = max(vals.min(), q1 - 1.5 * iqr)
        whisker_hi = min(vals.max(), q3 + 1.5 * iqr)
        box_cx = i
        box_hw = 0.07

        ax.add_patch(
            plt.Rectangle(
                (box_cx - box_hw, q1),
                2 * box_hw,
                iqr,
                facecolor="white",
                edgecolor=color,
                lw=1.5,
                zorder=4,
            )
        )
        # Median line (full width of box)
        ax.plot(
            [box_cx - box_hw, box_cx + box_hw],
            [median, median],
            color=color,
            lw=1.0,
            zorder=5,
        )
        # Whiskers showing the full range within 1.5 IQR, with vertical line 
        ax.plot([box_cx, box_cx], [whisker_lo, q1], color=color, lw=1.0, zorder=4)
        ax.plot([box_cx, box_cx], [q3, whisker_hi], color=color, lw=1.0, zorder=4)
        # Horizontal caps at the ends of the whiskers
        ax.plot(
            [box_cx - box_hw / 2, box_cx + box_hw / 2],
            [whisker_lo, whisker_lo],
            color=color,
            lw=1.0,
            zorder=4,
        )
        ax.plot(
            [box_cx - box_hw / 2, box_cx + box_hw / 2],
            [whisker_hi, whisker_hi],
            color=color,
            lw=0.5,
            zorder=4,
        )

        # ── Half-violin (KDE, right side, overlaps right half of boxplot) ──
        kde = gaussian_kde(vals, bw_method="silverman")
        kde_y = np.linspace(vals.min(), vals.max(), 300)
        kde_x = kde(kde_y)
        kde_x_norm = kde_x / kde_x.max() * 0.35  # max width in data units
        ax.fill_betweenx(
            kde_y, i + 0.05, i + 0.05 + kde_x_norm, color=color, alpha=0.55
        )
        ax.plot(i + 0.05 + kde_x_norm, kde_y, color=color, lw=0.8)


    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.upper() for c in conditions])
    ax.set_xlim(-0.6, len(conditions) - 0.35)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", ls="-", lw=0.3, alpha=0.5)
    #ax.autoscale(enable=True, axis="y", tight=False)
    #ax.margins(y=0.1)
    #ax.axhline(0, color="grey", lw=0.4, alpha=0.5)

    if standalone:
        fig.suptitle(f"Raincloud: {ylabel} by condition", y=0.995)
        fig.tight_layout()
        if output_path:
            save_figure(fig, output_path)
        else:
            plt.show()


# ── Parameter plots ──────────────────────────────────────────────────────────

def _build_parameters_across_blocks_separate(
    results_df: pd.DataFrame,
    params: list[tuple[str, str]],
    conditions: list[str],
    block_ticks: list,
    participant_col: str,
    condition_col: str,
    block_col: str,
    plot_individual_traces: bool,
) -> plt.Figure:
    """One column per condition, one row per parameter."""
    n_params = len(params)
    n_cond = len(conditions)
    fig, axes = plt.subplots(
        n_params, n_cond,
        figsize=(2.5 * n_cond, 2.0 * n_params),
        sharex=True, squeeze=False,
    )
    for i_row in range(n_params):
        for i_col in range(1, n_cond):
            axes[i_row][i_col].sharey(axes[i_row][0])

    for i_param, (col, ylabel) in enumerate(params):
        for i_cond, cond in enumerate(conditions):
            ax = axes[i_param][i_cond]
            color = TRIAL_TYPE_COLORS.get(cond, "tab:blue")
            cond_data = results_df[results_df[condition_col] == cond].dropna(subset=[col])
            _plot_block_traces_with_mean_sem(
                ax, cond_data, col, block_col, participant_col,
                color, block_ticks, is_circular=(col in _CIRCULAR_COLS),
                plot_individual_traces=plot_individual_traces,
            )
            if i_param == 0:
                ax.set_title(cond.upper())
            if i_cond == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.tick_params(labelleft=False)
            if i_param == n_params - 1:
                ax.set_xlabel("Block")
    return fig


def _build_parameters_across_blocks_combined(
    results_df: pd.DataFrame,
    params: list[tuple[str, str]],
    conditions: list[str],
    block_ticks: list,
    participant_col: str,
    condition_col: str,
    block_col: str,
    plot_individual_traces: bool,
    show_distribution: bool = False,
) -> plt.Figure:
    """All conditions overlaid per row, one row per parameter.

    show_distribution=True adds a raincloud distribution column to the right.
    """
    n_params = len(params)
    n_cols = 2 if show_distribution else 1
    figsize = (8 if show_distribution else 5.0, 2.0 * n_params)
    gridspec_kw = {"width_ratios": [3, 2]} if show_distribution else {}

    fig, axes = plt.subplots(
        n_params, n_cols,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
        squeeze=False,
    )
    for i_row in range(1, n_params):
        axes[i_row][0].sharex(axes[0][0])
        if show_distribution:
            axes[i_row][1].sharex(axes[0][1])

    for i_param, (col, ylabel) in enumerate(params):
        ax_ts = axes[i_param][0]
        for cond in conditions:
            color = TRIAL_TYPE_COLORS.get(cond, "tab:blue")
            cond_data = results_df[results_df[condition_col] == cond].dropna(subset=[col])
            _plot_block_traces_with_mean_sem(
                ax_ts, cond_data, col, block_col, participant_col,
                color, block_ticks, is_circular=(col in _CIRCULAR_COLS),
                plot_individual_traces=plot_individual_traces,
                label=cond.upper(),
            )
        ax_ts.set_ylabel(ylabel)
        if i_param == n_params - 1:
            ax_ts.set_xlabel("Block")
        else:
            ax_ts.tick_params(labelbottom=False)

        if show_distribution:
            ax_rc = axes[i_param][1]
            ax_rc.sharey(ax_ts)
            plot_raincloud(
                results_df,
                parameter=col,
                parameter_label=ylabel,
                participant_col=participant_col,
                condition_col=condition_col,
                block_col=block_col,
                ax=ax_rc,
            )
            ax_rc.set_ylabel("")
            ax_rc.tick_params(labelleft=False)
            if i_param < n_params - 1:
                ax_rc.tick_params(labelbottom=False)

    axes[0][0].legend(loc="upper right", fontsize=7, frameon=False, ncols=2)
    return fig


def plot_parameters_across_blocks(
    results_df: pd.DataFrame,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    output_path: Path | None = None,
    plot_individual_traces: bool = False,
    combine_conditions: bool = False,
    show_distribution: bool = False,
) -> None:
    """Plot parameters across blocks, one row per outcome.

    combine_conditions=False (default): one column per condition.
    combine_conditions=True: all conditions overlaid in a single column.
    show_distribution=True: add raincloud column to the right (only with combine_conditions=True).
    mean +/- SEM overlay. Phase parameters use circular mean and SEM.
    """
    params = [
        ("baseline_mean", "Baseline mean [mm]"),
        ("tonic_min", "Tonic minimum [mm]"),
        # ("tonic_mean", "Tonic mean [mm]"),
        ("amplitude", "Amplitude f [mm]"),
        ("amplitude_2f", "Amplitude 2f [mm]"),
        ("phase_rad", "Phase f [rad]"),
        ("phase_2f_rad", "Phase 2f [rad]"),
    ]
    conditions = sorted(results_df[condition_col].dropna().unique())
    block_ticks = sorted(results_df[block_col].dropna().unique())

    shared = dict(
        results_df=results_df, params=params, conditions=conditions,
        block_ticks=block_ticks, participant_col=participant_col,
        condition_col=condition_col, block_col=block_col,
        plot_individual_traces=plot_individual_traces,
    )
    if combine_conditions:
        fig = _build_parameters_across_blocks_combined(**shared, show_distribution=show_distribution)
    else:
        fig = _build_parameters_across_blocks_separate(**shared)

    fig.suptitle("Parameter changes across conditions and blocks", y=0.995)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure in: {output_path}")
        if not _in_jupyter():
            plt.close(fig)
    else:
        plt.show()


RAINCLOUD_PARAMS: list[tuple[str, str]] = [
    ("baseline_mean", "Baseline mean [mm]"),
    #("tonic_min",     "Tonic minimum [mm]"),
    ("tonic_mean",    "Tonic mean [mm]"),
    ("amplitude",     "Amplitude f [mm]"),
    ("amplitude_2f",  "Amplitude 2f [mm]"),
    ("phase_rad",     "Phase f [rad]"),
    ("phase_2f_rad",  "Phase 2f [rad]"),
]


def plot_raincloud_grid(
    results_df: pd.DataFrame,
    params: list[tuple[str, str]] | None = None,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    output_path: Path | None = None,
) -> None:
    """Raincloud grid: one subplot per parameter, 2 columns × 3 rows.

    Parameters
    ----------
    params : list of (column, label) tuples, optional
        Parameters to plot.  Defaults to ``RAINCLOUD_PARAMS``.
    """
    if params is None:
        params = RAINCLOUD_PARAMS

    conditions = sorted(results_df[condition_col].dropna().unique())
    panel_w = 1 * len(conditions)
    n_cols, n_rows = 2, 3

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_w * n_cols, 3 * n_rows), squeeze=False)

    for idx, (param_col, param_label) in enumerate(params):
        ax = axes[idx // n_cols, idx % n_cols]
        plot_raincloud(
            results_df,
            parameter=param_col,
            parameter_label=param_label,
            condition_col=condition_col,
            block_col=block_col,
            ax=ax,
        )
        ax.set_title(param_label)

    fig.suptitle("Raincloud plots: cosinor parameters by condition", y=1.001)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure in: {output_path}")
        if not _in_jupyter():
            plt.close(fig)
    else:
        plt.show()


def plot_parameter_changes_raincloud(
    results_df: pd.DataFrame,
    parameter: str,
    parameter_label: str | None = None,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    output_path: Path | None = None,
) -> None:
    """2×3 grid for one parameter: 5 per-condition dynamics panels + raincloud.

    Layout (row × col):
        [cond 0] [cond 1] [cond 2]
        [cond 3] [cond 4] [raincloud]

    Each dynamics panel shows individual-participant traces and mean ± SEM
    across blocks.  Phase parameters use circular mean / SEM.
    """
    ylabel = parameter_label or parameter
    conditions = sorted(results_df[condition_col].dropna().unique())
    block_ticks = sorted(results_df[block_col].dropna().unique())
    is_circular = parameter in _CIRCULAR_COLS

    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.0 * n_rows), squeeze=False)

    # Share y-axis across the 5 dynamics panels
    ax_ref = axes[0, 0]
    for idx in range(1, len(conditions)):
        axes[idx // n_cols, idx % n_cols].sharey(ax_ref)

    for idx, cond in enumerate(conditions):
        ax = axes[idx // n_cols, idx % n_cols]
        color = TRIAL_TYPE_COLORS.get(cond, "tab:blue")
        cond_data = results_df[results_df[condition_col] == cond].dropna(subset=[parameter])

        _plot_block_traces_with_mean_sem(
            ax, cond_data, parameter, block_col, participant_col,
            color, block_ticks, is_circular=is_circular,
        )
        ax.set_title(cond.upper())
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Block")
        else:
            ax.tick_params(labelbottom=False)
        if idx % n_cols == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.tick_params(labelleft=False)

    # Panel 5 (row 1, col 2): raincloud
    ax_rain = axes[1, 2]
    plot_raincloud(
        results_df,
        parameter=parameter,
        parameter_label=ylabel,
        participant_col=participant_col,
        condition_col=condition_col,
        block_col=block_col,
        ax=ax_rain,
    )
    ax_rain.set_ylim(ax_ref.get_ylim())
    ax_rain.set_ylabel("")
    ax_rain.set_yticklabels([])

    fig.suptitle(ylabel, y=1.002)
    fig.tight_layout(h_pad=0.4, w_pad=0.4)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure in: {output_path}")
        if not _in_jupyter():
            plt.close(fig)
    else:
        plt.show()


# ── Decomposition plots ──────────────────────────────────────────────────────

def compute_grand_average(
    df: pd.DataFrame,
    condition_col: str = DEFAULT_CONDITION_COL,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    time_col: str = DEFAULT_TIME_COL,
    outcome_col: str = DEFAULT_OUTCOME_COL,
) -> pd.DataFrame:
    """Hierarchical grand average per condition (blocks → participants)."""
    block_avg = (
        df.groupby([condition_col, participant_col, time_col], dropna=False)[outcome_col]
        .mean()
        .reset_index()
    )
    result_grand = (
        block_avg.groupby([condition_col, time_col], dropna=False)[outcome_col]
        .mean()
        .reset_index()
        .rename(columns={outcome_col: "grand_" + outcome_col})
    )
    return result_grand


def plot_grand_average_decomposition(
    grand_averages: pd.DataFrame,
    outcome_col: str = DEFAULT_OUTCOME_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    time_col: str = DEFAULT_TIME_COL,
    cutoff: float = DEFAULT_CUTOFF_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
    freq: float = DEFAULT_FREQUENCY,
    t_stim_start: float = DEFAULT_T_STIM_START,
    t_stim_end: float = DEFAULT_T_STIM_END,
    add_poststim_sec: float = DEFAULT_ADD_POSTSTIM_SEC,
    output_path: Path | None = None,
) -> None:
    """Plot two-step decomposition on grand averages (diagnostic).

    For each condition: left = data + tonic, right = residual + cosinor fit.
    """
    conditions = sorted(grand_averages[condition_col].dropna().unique())
    n_cond = len(conditions)

    fig, axes = plt.subplots(
        n_cond, 2, figsize=(9, 2.2 * n_cond),
        sharex="col", squeeze=False,
        sharey="col",
    )

    for i, cond in enumerate(conditions):
        cond_df = grand_averages[grand_averages[condition_col] == cond]
        t = cond_df[time_col].values.astype(float)
        y = cond_df[outcome_col].values.astype(float)
        color = TRIAL_TYPE_COLORS.get(cond, "tab:blue")

        tonic_out = extract_tonic_and_residual(
            t, y, cutoff=cutoff, order=filter_order,
            t_stim_start=t_stim_start, t_stim_end=t_stim_end,
        )
        tonic = tonic_out["tonic"]
        residual = tonic_out["residual"]

        cosinor = fit_cosinor_residual(
            t, residual, freq=freq,
            t_stim_start=t_stim_start,
            t_stim_end=(t_stim_end + add_poststim_sec),
        )

        ax_left = axes[i, 0]
        ax_left.axvspan(t_stim_start, t_stim_end, color="grey", alpha=0.10, linewidth=0)
        ax_left.axhline(0, color="grey", lw=0.4, alpha=0.5)
        ax_left.plot(t, y, color="grey", lw=0.8, alpha=0.6, label="Raw pupil trace")
        ax_left.plot(t, tonic, color=color, lw=1.8, ls="--", label="Tonic (low-pass)")

        if not np.isnan(tonic_out["tonic_min"]):
            secs_to_min = float(tonic_out["tonic_min_time"]) - float(t_stim_start)
            ax_left.plot(
                tonic_out["tonic_min_time"], tonic_out["tonic_min"],
                "v", color="red", ms=7, zorder=5,
                label=f"Min = {tonic_out['tonic_min']:.3f} mm\nLatency: {secs_to_min:.2f} s",
            )

        ax_left.set_ylabel(f"{cond.upper()}\nRel. size [mm]")
        ax_left.legend(loc="lower right", frameon=False)
        ax_left.xaxis.set_major_locator(MultipleLocator(5))
        ax_left.xaxis.set_minor_locator(MultipleLocator(1))
        ax_left.tick_params(axis="x", which="major", length=4)
        ax_left.tick_params(axis="x", which="minor", length=2)
        if i == 0:
            ax_left.set_title("Step 1: Low pass tonic fit on empirical data")
        if i == n_cond - 1:
            ax_left.set_xlabel("Trial time [s]")

        ax_right = axes[i, 1]
        ax_right.axvspan(t_stim_start, t_stim_end, color="grey", alpha=0.10, linewidth=0)
        ax_right.axhline(0, color="grey", lw=0.4, alpha=0.5)
        ax_right.plot(t, residual, color="silver", lw=0.6, label="Residual")

        if not np.isnan(cosinor["alpha_sin"]):
            fit_mask = (t >= t_stim_start) & (t <= t_stim_end + add_poststim_sec)
            t_fit = t[fit_mask]
            omega = 2.0 * np.pi * freq
            y_cos = (
                cosinor["beta_intercept"]
                + cosinor["alpha_sin"] * np.sin(omega * t_fit)
                + cosinor["beta_cos"] * np.cos(omega * t_fit)
                + cosinor["alpha_sin_2f"] * np.sin(2.0 * omega * t_fit)
                + cosinor["beta_cos_2f"] * np.cos(2.0 * omega * t_fit)
            )
            ax_right.plot(
                t_fit, y_cos, color=color, lw=1.5,
                label=r"Cosinor ($A_f$ =" + f"{cosinor['amplitude']:.3f} mm, \n"
                      + r"$A_{2f}$ =" + f"{cosinor['amplitude_2f']:.3f} mm, "
                      + r"$R^2$ =" + f"{cosinor['r_squared']:.3f})",
            )

        ax_right.set_ylabel("")
        ax_right.legend(loc="lower right", frameon=False)
        ax_right.xaxis.set_major_locator(MultipleLocator(5))
        ax_right.xaxis.set_minor_locator(MultipleLocator(1))
        ax_right.tick_params(axis="x", which="major", length=4)
        ax_right.tick_params(axis="x", which="minor", length=2)
        if i == 0:
            ax_right.set_title("Step 2: Cosinor fit on residuals")
        if i == n_cond - 1:
            ax_right.set_xlabel("Trial time [s]")

    fig.suptitle("Two-step decomposition on pupil traces", y=0.995)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure in: {output_path}")
        if not _in_jupyter():
            plt.close(fig)
    else:
        plt.show()


def plot_2step_fits_with_decomposition_overlay(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    outcome_col: str = DEFAULT_OUTCOME_COL,
    time_col: str = DEFAULT_TIME_COL,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    cutoff: float = DEFAULT_CUTOFF_HZ,
    filter_order: int = DEFAULT_FILTER_ORDER,
    freq: float = DEFAULT_FREQUENCY,
    t_stim_start: float = DEFAULT_T_STIM_START,
    t_stim_end: float = DEFAULT_T_STIM_END,
    add_poststim_sec: float = DEFAULT_ADD_POSTSTIM_SEC,
    participant: int | str | None = None,
    output_path: Path | None = None,
) -> None:
    """Plot example two-step fits for one participant (conditions × blocks).

    Shows original data, tonic (low-pass), and tonic + cosinor reconstruction.
    """
    if participant is None:
        median_r2 = results_df.groupby(participant_col)["r_squared"].median()
        participant = median_r2.sort_values().index[len(median_r2) // 2]

    conditions = sorted(results_df[condition_col].dropna().unique())
    blocks = sorted(results_df[block_col].dropna().unique())
    n_rows = len(conditions)
    n_cols = len(blocks)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2.0 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )

    omega = 2.0 * np.pi * freq

    for i_cond, cond in enumerate(conditions):
        color = TRIAL_TYPE_COLORS.get(cond, "tab:blue")
        for i_block, block in enumerate(blocks):
            ax = axes[i_cond][i_block]

            mask = (
                (df[participant_col] == participant)
                & (df[condition_col] == cond)
                & (df[block_col] == block)
            )
            subset = df[mask].sort_values(time_col)

            show_legend = (i_block == len(blocks) - 1)

            if len(subset) < 10:
                ax.text(
                    0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="grey",
                )
            else:
                t = subset[time_col].values.astype(float)
                y = subset[outcome_col].values.astype(float)

                tonic_out = extract_tonic_and_residual(
                    t, y, cutoff=cutoff, order=filter_order,
                    t_stim_start=t_stim_start, t_stim_end=t_stim_end,
                )

                l1, = ax.plot(t, y, color=color, lw=0.5, alpha=0.4, label="Empirical data")
                l2, = ax.plot(t, tonic_out["tonic"], color="black", lw=0.8, ls="--", alpha=0.7, label="Tonic fit")

                fit_mask = (t >= t_stim_start) & (t <= t_stim_end + add_poststim_sec)
                fit_row = results_df[
                    (results_df[participant_col] == participant)
                    & (results_df[condition_col] == cond)
                    & (results_df[block_col] == block)
                ]
                if not fit_row.empty and not pd.isna(fit_row["alpha_sin"].iloc[0]):
                    t_stim = t[fit_mask]
                    b0 = fit_row["beta_intercept"].iloc[0]
                    alpha_sin = fit_row["alpha_sin"].iloc[0]
                    beta_cos = fit_row["beta_cos"].iloc[0]
                    alpha_sin_2f = fit_row["alpha_sin_2f"].iloc[0]
                    beta_cos_2f = fit_row["beta_cos_2f"].iloc[0]
                    cosinor_curve = (
                        b0
                        + alpha_sin * np.sin(omega * t_stim)
                        + beta_cos * np.cos(omega * t_stim)
                        + alpha_sin_2f * np.sin(2.0 * omega * t_stim)
                        + beta_cos_2f * np.cos(2.0 * omega * t_stim)
                    )
                    reconstruction = tonic_out["tonic"][fit_mask] + cosinor_curve
                    l3, = ax.plot(t_stim, reconstruction, color=color, lw=1.0, alpha=0.9, label="Tonic + Cosinor fit")
                else:
                    l3 = None

            ax.axvspan(t_stim_start, t_stim_end, color="grey", alpha=0.08, linewidth=0)
            ax.axhline(0, color="grey", lw=0.3, alpha=0.4)

            # Titles and labels only on the leftmost and top panels
            if i_cond == 0:
                ax.set_title(f"Block {block}")
            if i_block == 0:
                if outcome_col == "rel_size_mm":
                    ax.set_ylabel("Rel. size [mm]")
                else:
                    ax.set_ylabel(f"{cond.upper()}")
            if i_cond == n_rows - 1:
                ax.set_xlabel("t (s)")
            ax.tick_params()
            
            # Legend only on the last column of each row
            if show_legend:
                handles = [h for h in [l1, l2, l3] if h is not None]
                ax.legend(
                    handles=handles,
                    loc="center left",
                    bbox_to_anchor=(1.01, 0.5),
                    frameon=False,
                )

    fig.suptitle(
        f"Participant {participant}: two-step decomposition (tonic and transient response)",
        y=0.995,
    )
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure in: {output_path}")
        if not _in_jupyter():
            plt.close(fig)
    else:
        plt.show()


# ── Plotting API ──────────────────────────────────────────────────────

def parameter_plots(
    results_csv: Path | None = None, output_dir: Path | None = None
) -> None:
    """
    Generate summary plots from the fitted cosinor model parameters.

    This function loads the results CSV produced by the two-step cosinor fitting
    and generates a set of diagnostic and summary figures based solely on the fitted
    parameters (no raw pupil data required). Plots include parameter trajectories,
    raincloud plots, and exemplary participant summaries.

    Parameters
    ----------
    results_csv : Path or None, optional
        Path to the results CSV file containing fitted cosinor parameters. If None,
        uses the default path from the pipeline.
    output_dir : Path or None, optional
        Output directory root. Figures will be saved to <output_dir>/figures/pupil_cosinor/.
        If None, uses the default results directory.

    Returns
    -------
    None
        Figures are saved to disk; nothing is returned.
    """
    if results_csv is None:
        results_csv = DEFAULT_RESULTS_CSV
    if output_dir is None:
        output_dir = DEFAULT_RESULTS_DIR

    if not results_csv.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {results_csv}\n"
            "Run fit_pupil_trace_2step_cosinor_model first."
        )

    results_df = pd.read_csv(results_csv)
    print(f"Loaded results: {len(results_df)} rows from {results_csv}")

    fig_dir = output_dir / "figures" / "pupil_cosinor"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Parameter plots (results CSV only) ───────────────────────────────────
    plot_parameters_across_blocks(
        results_df,
        plot_individual_traces=False,
        output_path=fig_dir / "fig_pupil_cosinor_parameters_single.png",
    )
    plot_parameters_across_blocks(
        results_df,
        plot_individual_traces=False,
        combine_conditions=True,
        show_distribution=True,
        output_path=fig_dir / "fig_pupil_cosinor_parameters.png",
    )
    plot_raincloud_grid(
        results_df,
        output_path=fig_dir / "fig_raincloud_parameters.png",
    )
    for param_col, param_label in RAINCLOUD_PARAMS:
        plot_parameter_changes_raincloud(
            results_df,
            parameter=param_col,
            parameter_label=param_label,
            output_path=fig_dir / f"fig_param_changes_{param_col}.png",
        )

    exemplary_df = results_df[results_df[DEFAULT_PARTICIPANT_COL] == EXEMPLARY_PARTICIPANT]
    if exemplary_df.empty:
        raise ValueError(f"Exemplary participant {EXEMPLARY_PARTICIPANT} not in {results_csv}")
    plot_parameters_across_blocks(
        exemplary_df,
        output_path=fig_dir
        / f"fig_pupil_cosinor_parameters_{EXEMPLARY_PARTICIPANT}.png",
    )


def decomposition_plots(
    input_data: Path | None = None,
    results_df: pd.DataFrame | None = None,
    plot_individual_participants: bool = False,
    output_dir: Path | None = None,
    load_settings_json: Path | None = None,
) -> None:
    """
    Generate decomposition plots overlaying model fits on raw pupil traces.

    This function loads the aggregated pupil data and fitted cosinor parameters,
    then generates plots that decompose the raw pupil traces into tonic and oscillatory
    components, overlaying the model fit for both grand-average and exemplary participants.
    Optionally, per-participant plots can be generated.

    Parameters
    ----------
    input_data : Path, required
        Path to the aggregated pupil data CSV (output of aggregation step).
    results_df : pd.DataFrame or None, optional
        DataFrame of fitted cosinor parameters. If None, loads from default results CSV.
    plot_individual_participants : bool, optional
        If True, generate decomposition plots for each participant (slow).
    output_dir : Path or None, optional
        Output directory root. Figures will be saved to <output_dir>/figures/pupil_cosinor/.
        If None, uses the default results directory.
    load_settings_json : Path or None, optional
        If provided, loads all settings from this JSON file and overrides other arguments.
            cutoff : float, optional
                Low-pass filter cutoff frequency in Hz (default: 0.2).
            filter_order : int, optional
                Butterworth filter order (default: 2).
            freq : float, optional
                Stimulus frequency in Hz (default: 0.4).
            t_stim_start : float, optional
                Start time of stimulus window in seconds (default: 0).
            t_stim_end : float, optional
                End time of stimulus window in seconds (default: 5).
            add_poststim_sec : float, optional
                Post-stimulus buffer in seconds to append to fit window (default: 0;
                set via DEFAULT_ADD_POSTSTIM_SEC in analysis_io.py).
            outcome_col : str, optional
                Name of the outcome variable column (default: 'rel_size_mm').
        Path to the JSON file written by fit_pupil_trace_2step_cosinor_model.
        Defaults to DEFAULT_SETTINGS_JSON (results/metadata/pupil_cosinor_2step_settings.json).

    Returns
    -------
    None
        Figures are saved to disk; nothing is returned.
    """
    if output_dir is None:
        output_dir = DEFAULT_RESULTS_DIR
    fig_dir = output_dir / "figures" / "pupil_cosinor"
    fig_dir.mkdir(parents=False, exist_ok=True)

    # Load aggregated pupil data (defaults to data/aggregated/data.csv)
    input_df = load_aggregated_data(input_data)

    # Load results DataFrame if not passed in directly
    if results_df is None:
        if not DEFAULT_RESULTS_CSV.exists():
            raise FileNotFoundError(
                f"Results CSV not found: {DEFAULT_RESULTS_CSV}\n"
                "Run fit_pupil_trace_2step_cosinor_model first."
            )
        results_df = pd.read_csv(DEFAULT_RESULTS_CSV)
        print(f"Loaded results: {len(results_df)} rows from {DEFAULT_RESULTS_CSV}")

    exemplary_participant = EXEMPLARY_PARTICIPANT

    # Load decomposition settings from JSON (must match what the fitter used)
    settings_path = load_settings_json or DEFAULT_SETTINGS_JSON
    with open(settings_path, "r") as f:
        loaded = json.load(f)
        cutoff = loaded["cutoff"]
        filter_order = loaded["filter_order"]
        freq = loaded["freq"]
        t_stim_start = loaded["t_stim_start"]
        t_stim_end = loaded["t_stim_end"]
        add_poststim_sec = loaded["add_poststim_sec"]
        outcome_col = loaded["outcome_col"]

    # ── Decomposition plots ───────────────────────────────────────────────────
    grand_averages = compute_grand_average(
        input_df,
        condition_col=DEFAULT_CONDITION_COL,
        participant_col=DEFAULT_PARTICIPANT_COL,
        outcome_col=outcome_col,
    )
    grand_averages[outcome_col] = grand_averages[f"grand_{outcome_col}"].copy()

    plot_grand_average_decomposition(
        grand_averages,
        outcome_col=outcome_col,
        cutoff=cutoff,
        filter_order=filter_order,
        freq=freq,
        t_stim_start=t_stim_start,
        t_stim_end=t_stim_end,
        add_poststim_sec=add_poststim_sec,
        output_path=fig_dir / "fig_pupil_cosinor_decomposition_average.png",
    )

    df_exemplary = input_df[
        (input_df[DEFAULT_PARTICIPANT_COL] == exemplary_participant)
        & (input_df[DEFAULT_BLOCK_COL] == 3)
    ].copy()
    plot_grand_average_decomposition(
        df_exemplary,
        outcome_col=outcome_col,
        cutoff=cutoff,
        filter_order=filter_order,
        freq=freq,
        t_stim_start=t_stim_start,
        t_stim_end=t_stim_end,
        add_poststim_sec=add_poststim_sec,
        output_path=fig_dir
        / f"fig_pupil_cosinor_decomposition_{exemplary_participant}.png",
    )

    if plot_individual_participants:
        for participant in sorted(input_df[DEFAULT_PARTICIPANT_COL].dropna().unique()):
            (fig_dir / f"participant_{participant}").mkdir(parents=True, exist_ok=True)
            plot_2step_fits_with_decomposition_overlay(
                df=input_df,
                results_df=results_df,
                outcome_col=outcome_col,
                cutoff=cutoff,
                filter_order=filter_order,
                freq=freq,
                t_stim_start=t_stim_start,
                t_stim_end=t_stim_end,
                add_poststim_sec=add_poststim_sec,
                participant=participant,
                output_path=fig_dir
                / f"participant_{participant}"
                / f"fig_cosinor_2step_fits_with_overlay_{participant}.png",
            )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate pupil cosinor figures from the fitted results CSV.",
    )
    parser.add_argument(
        "--results-csv", type=Path, default=None,
        help=f"Path to results CSV (default: {DEFAULT_RESULTS_CSV}).",
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to aggregated data CSV (default: data/aggregated/data.csv).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output root directory (default: results/).",
    )
    parser.add_argument(
        "--plot-individual", action="store_true",
        help="Generate per-participant decomposition plots (slow).",
    )
    parser.add_argument(
        "--load-settings-json", type=Path, default=None,
        help="If provided, load decomposition settings from this JSON file.",
    )
    args = parser.parse_args()

    parameter_plots(
        results_csv=args.results_csv,
        output_dir=args.output_dir,
    )
    decomposition_plots(
        input_data=args.data,
        output_dir=args.output_dir,
        plot_individual_participants=args.plot_individual,
        load_settings_json=args.load_settings_json,
    )
