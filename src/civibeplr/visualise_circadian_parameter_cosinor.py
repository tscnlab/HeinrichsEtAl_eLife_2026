"""visualise_circadian_parameter_cosinor.py

Post-hoc visualisation of M1 circadian cosinor fits for the paper.

Reads per-condition *_predictions.csv files saved by
analyse_circadian_pupil_parameters.py --fit and produces
fitted-vs-empirical figures.

Figures
-------
fig_fitted_vs_empirical_{condition}.png   (layout "per_condition")
    One figure per condition: per-participant grid (observed values with
    conditional posterior mean and 95 % ETI, fixed + random effects) plus a
    wide group-average panel (block means ± SE with marginal posterior mean
    and 95 % ETI, fixed effects only).

fig_fitted_vs_empirical_all_conditions.png   (layout "combined")
    All conditions in a single figure. One compact panel per condition showing
    the group-level fit (marginal posterior mean + 95 % ETI), the observed
    block means ± SE, and one thin line per participant so individual
    variability is visible without a per-participant grid. That line is the
    conditional posterior mean by default, or the participant's empirical
    values as tiny connected points with --individual-style observed.
    The final panel is a forest plot comparing the circadian amplitude estimate
    and the Bayes factor across conditions.

Usage
-----
python -m civibeplr.visualise_circadian_parameter_cosinor [--outcome amplitude tonic_min]
                                                          [--layout combined]
                                                          [--share-y]
                                                          [--individual-style observed]
"""


# Standard library imports
from __future__ import annotations
import re
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

# Project-specific imports
from civibeplr.analyse_circadian_pupil_parameters import DEFAULT_IDATA_DIR
from civibeplr.analysis_io import (
    ANALYSIS_PARTICIPANT_COL,
    PUPIL_BLOCK_COL,
    PUPIL_CONDITION_COL,
    PARAMETER_CHOICES,
    get_parameter_label,
    lowercase_first_word,
)
from civibeplr.vis_io import (
    _in_jupyter,
    apply_font_style,
    readable,
    CONDITION_LABELS,
    CONDITION_ORDER,
    TRIAL_TYPE_COLORS,
)

apply_font_style()

# Project root and default output directory for figures
_PROJECT_DIR        = Path(__file__).resolve().parents[2]
DEFAULT_FIGURES_DIR = _PROJECT_DIR / "results" / "figures" / "circadian_cosinor"

# Used when a condition name does not match any key in TRIAL_TYPE_COLORS
_FALLBACK_COLOR = "#787878"


def _condition_color(condition: str) -> str:
    """Return the TRIAL_TYPE_COLORS entry for *condition*, falling back to grey."""
    key = condition.lower().replace(" ", "_")
    for color_key in sorted(TRIAL_TYPE_COLORS, key=len, reverse=True):
        if color_key in key:  # or key.startswith(color_key)
            return TRIAL_TYPE_COLORS[color_key]
    return _FALLBACK_COLOR


def _condition_label(condition: str) -> str:
    """Return the long condition name, falling back to the upper-cased key."""
    return CONDITION_LABELS.get(condition.lower().replace(" ", "_"), condition.upper())


def _summary_value(summary_df: pd.DataFrame | None, condition: str, column: str) -> float | None:
    """Return *column* for *condition* from the all-conditions summary table.

    Returns None when the table, the condition row, the column, or the value
    itself is missing, so callers can simply skip the annotation.
    """
    if summary_df is None or column not in summary_df.columns:
        return None
    cond_key = condition.lower().replace(" ", "_")
    keys = summary_df["condition"].str.lower().str.replace(" ", "_")
    row = summary_df[keys == cond_key]
    if row.empty or pd.isna(row[column].iloc[0]):
        return None
    return float(row[column].iloc[0])


def _unit_from_label(outcome_col: str) -> str:
    """Extract the bracketed unit (e.g. 'mm') from a parameter axis label."""
    match = re.search(r"\[([^\]]*)\]", get_parameter_label(outcome_col))
    return match.group(1) if match else ""


def _fmt_bayes_factor(log_bf10: float) -> str:
    """Format log10 BF10 as the Bayes factor favouring the better-supported model.

    Positive log BF10 is reported as BF10, negative as BF01, so the printed
    value is always >= 1 and reads as "evidence for M1 / for M0".
    """
    label  = "10" if log_bf10 >= 0 else "01"
    log_bf = abs(log_bf10)
    value  = 10 ** log_bf
    if value >= 1000:  # switch to scientific notation before the digits run away
        exponent = int(np.floor(log_bf))
        mantissa = value / 10 ** exponent
        number   = f"{mantissa:.1f}$\\times$10$^{{{exponent}}}$"
    elif value >= 10:
        number = f"{value:.0f}"
    else:
        number = f"{value:.2f}"
    return f"BF$_{{{label}}}$ = {number}"


def _build_block_average(preds_df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    """Aggregate per-block observed mean/SE and marginal posterior columns."""
    # Per-block observed statistics from the raw predictions
    block_obs = (
        preds_df.groupby(PUPIL_BLOCK_COL)[outcome_col]
        .agg(["mean", "sem"])
        .reset_index()
        .rename(columns={"mean": "obs_mean", "sem": "obs_se"})
    )
    # Per-block marginal posterior summaries (fixed effects only, averaged over participants)
    block_fit = (
        preds_df.groupby(PUPIL_BLOCK_COL)[
            ["hours_since_start", "mu_marg_mean", "mu_marg_lo", "mu_marg_hi"]
        ]
        .mean()
        .reset_index()
    )
    return block_obs.merge(block_fit, on=PUPIL_BLOCK_COL).sort_values(PUPIL_BLOCK_COL)


def _draw_average_panel(
    ax: plt.Axes,
    block: pd.DataFrame,
    outcome_col: str,
    condition: str = "flux",
    *,
    compact: bool = False,
    amplitude: float | None = None,
    log_bf10: float | None = None,
    show_stats: bool = True,
) -> None:
    """Draw group-average fixed-effect panel onto *ax*.

    Parameters
    ----------
    amplitude : float or None
        Circadian amplitude estimate to annotate in the upper-right corner.
    log_bf10 : float or None
        Log10 Bayes Factor (log BF10) to annotate.
    show_stats : bool
        Whether to render the amplitude/BF annotation.
    """
    x = block[PUPIL_BLOCK_COL].values
    fs = 7 if compact else 10

    # Observed block means with SE error bars
    ax.errorbar(
        x, block["obs_mean"].values, yerr=block["obs_se"].values,
        fmt="o", color=_condition_color(condition), capsize=3, ms=4 if compact else 6,
        label="Observed (mean ± SE)",
    )
    # Marginal posterior mean (fixed effects only, no random participant offsets)
    ax.plot(x, block["mu_marg_mean"].values, color="black",
            lw=1.5 if compact else 2, label="M1 marginal mean")
    # 95 % ETI shading around the marginal posterior mean
    ax.fill_between(
        x, block["mu_marg_lo"].values, block["mu_marg_hi"].values,
        color=_FALLBACK_COLOR, alpha=0.20, linewidth=0, label="95 % ETI",
    )
    ax.set_xlabel(get_parameter_label("hours_since_start"))
    xticks_labels = np.arange(0, ((max(x) + 1) * 3.75), (4))
    xticks_labels = [int(tick) for tick in xticks_labels]
    ax.set_xticks(np.arange(0, max(x) + 1, (4/3.75)), xticks_labels)  # show all block values as x-ticks

    # make y-ticks 2 decimal places for consistent alignment without changing sign or magnitude of tick labels
    ax.set_yticks(ax.get_yticks(), [f"{tick:.2f}" for tick in ax.get_yticks()])
    ax.set_ylabel(get_parameter_label(outcome_col))
    ax.legend(frameon=False)

    # Optional annotation: amplitude and/or BF in the upper-right corner
    if show_stats:
        annotation_parts: list[str] = []
        if amplitude is not None:
            annotation_parts.append(f"Amplitude = {amplitude:.4f}")
        if log_bf10 is not None:
            bf10 = 10 ** log_bf10
            bf01 = 10 ** (-log_bf10)
            annotation_parts.append(f"BF$_{{10}}$ = {bf10:.3f}")
            annotation_parts.append(f"BF$_{{01}}$ = {bf01:.3f}")
        if annotation_parts:
            legend = ax.get_legend()
            legend_fs = legend.get_texts()[0].get_fontsize() if legend and legend.get_texts() else plt.rcParams["legend.fontsize"]
            ax.text(
                0.98, 0.97, "\n".join(annotation_parts),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=legend_fs, color="black",
            )


def plot_fitted_vs_empirical(
    preds_df: pd.DataFrame,
    condition: str,
    outcome_col: str = "amplitude",
    figures_dir: Path | None = None,
    *,
    show_stats: bool = True,
    summary_df: pd.DataFrame | None = None,
) -> None:
    """
    Plot M1 posterior predictions overlaid on observed data for a given condition.

    Produces one combined figure with:
    - Per-participant subplots: observed scatter + conditional posterior mean and 95 % ETI
      (fixed + random effects).
    - A highlighted "Group average" panel: block means +/- SE + marginal posterior mean
      and 95 % ETI (fixed effects only), visually distinguished by a shaded background.

    Parameters
    ----------
    preds_df : pd.DataFrame
        DataFrame loaded from *_predictions.csv; must contain participant_id, Block,
        hours_since_start, {outcome_col}, mu_cond_mean/lo/hi, mu_marg_mean/lo/hi.
    condition : str
        Name of the stimulus condition (e.g., 'Melanopsin').
    outcome_col : str, default 'amplitude'
        Outcome variable to plot (e.g., 'amplitude').
    figures_dir : Path or None
        Directory to save figures. If None, uses default.
    show_stats : bool, default True
        Annotate the group-average panel with amplitude and log BF10.
    summary_df : pd.DataFrame or None
        Row from circadian_pupil_all_conditions.csv for this condition. If None,
        annotations are skipped even when show_stats is True.
    """

    if figures_dir is None:
        figures_dir = DEFAULT_FIGURES_DIR / outcome_col
    figures_dir.mkdir(parents=True, exist_ok=True)

    cond_safe    = condition.replace(" ", "_").lower()
    participants = sorted(preds_df[ANALYSIS_PARTICIPANT_COL].unique())
    n_part       = len(participants)

    # ── Layout constants ────────────────────────────────────────────
    # Fixed 3×3 grid for up to 9 participants; slot 8 (bottom-right) is reserved
    # for the shared legend, so the pipeline supports up to 8 participants here.
    PART_COLS  = 3
    PART_ROWS  = 3
    PART_COLOR = _condition_color(condition)
    AVG_COLOR  = PART_COLOR  # reuse condition colour for both frames

    # Add a spacer row between participant grid and group average
    # Grid: 3 (participants) + 1 (spacer) + 1 (average) = 5 rows
    SPACER_HEIGHT = 0.2 # Adjust for desired gap size
    AVG_HEIGHT = 2
    fig = plt.figure(figsize=(3 * PART_COLS, 2.5 * PART_ROWS + 4))
    fig.suptitle(
        f"{condition.upper()}\nM1 fitted vs. observed {lowercase_first_word(get_parameter_label(outcome_col, title=True))}"#, fontsize=12, y=1.0,
    )

    gs = fig.add_gridspec(
        PART_ROWS + 2, PART_COLS,  # 3 participant rows + 1 spacer + 1 average
        height_ratios=[1] * PART_ROWS + [SPACER_HEIGHT, AVG_HEIGHT],
        hspace=0.4, wspace=0.25,  # smaller hspace for most rows
    )

    # Create one  axis per participant slot, then a single wide axis for the average
    part_axes = [
        fig.add_subplot(gs[r, c])
        for r in range(PART_ROWS)
        for c in range(PART_COLS)
    ]
    # avg_ax is in the last row (row index PART_ROWS + 1)
    avg_ax = fig.add_subplot(gs[PART_ROWS + 1, :])  # spans all columns

    # Section headings placed as figure-level text after tight_layout finalises positions
    _PART_HEADING = "Individual participants  (fixed + random effects)"
    _AVG_HEADING  = "Group average  (fixed effects only)"

    # ── Per-participant subplots ────────────────────────────────────
    y_label = get_parameter_label(outcome_col)
    x_label = get_parameter_label("hours_since_start")

    for idx, pid in enumerate(participants):
        ax  = part_axes[idx]
        sub = preds_df[preds_df[ANALYSIS_PARTICIPANT_COL] == pid].sort_values("hours_since_start")
        x   = sub["hours_since_start"].values

        # Raw observations
        ax.scatter(x, sub[outcome_col].values,
                   color=_condition_color(condition), s=20, zorder=3, label="Observed")
        # Conditional posterior mean (fixed + random effects)
        ax.plot(x, sub["mu_cond_mean"].values,
                color="black", lw=1.5, zorder=4, label="Posterior mean")
        # 95 % ETI shading for the conditional posterior
        ax.fill_between(
            x, sub["mu_cond_lo"].values, sub["mu_cond_hi"].values,
            color=_FALLBACK_COLOR, alpha=0.20, linewidth=0, label="95 % ETI",
        )
        ax.set_title(str(pid))#, fontsize=9)
        # Only label x-axis on bottom row to avoid clutter; share x-axis for vertical alignment
        if idx < PART_COLS * (PART_ROWS - 1):
            ax.set_xlabel("")  # hide x-axis label for non-bottom rows
        else:   
            ax.set_xlabel(x_label)#, fontsize=8)
        # set x-ticks to every 5 hours for better readability
        ax.set_xlim(-1, max(x) + 1)
        x_ticks = np.arange(0, max(x) + 1, 4)
        ax.set_xticks(x_ticks)
        if idx % PART_COLS == 0:
            ax.set_ylabel(y_label)#, fontsize=8)

    # Hide any unused participant slots; keep the last slot for the shared legend
    legend_slot = PART_ROWS * PART_COLS - 1  # index 8, bottom-right of grid
    for idx in range(n_part, PART_ROWS * PART_COLS):
        if idx != legend_slot:
            part_axes[idx].set_visible(False)

    # ── Extract amplitude / BF from summary for annotation ──────────
    _amplitude: float | None = None
    _log_bf10:  float | None = None
    if show_stats:
        _amplitude = _summary_value(summary_df, condition, "amplitude")
        _log_bf10  = _summary_value(summary_df, condition, "log_bf10")

    # ── Group-average panel ─────────────────────────────────────────
    block = _build_block_average(preds_df, outcome_col)
    _draw_average_panel(avg_ax, block, outcome_col, condition, compact=False,
                        amplitude=_amplitude, log_bf10=_log_bf10, show_stats=show_stats)

    # ── Shared legend in the 9th grid slot ─────────────────────────
    # Reuse handles from the first participant panel so the legend reflects all layers
    handles, labels = part_axes[0].get_legend_handles_labels()
    legend_ax = part_axes[legend_slot]
    legend_ax.set_visible(True)
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", frameon=False) #, framealpha=0.9) #, fontsize=9)

    #fig.tight_layout()

    # ── Coloured frames + headings (drawn after tight_layout) ───────
    # tight_layout must run first so axis positions are finalised before we compute
    # bounding boxes; canvas.draw() forces text extents to be resolved as well.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    PAD = 0.01  # uniform padding around each frame in figure-fraction units

    def _tight_bounds(axes_list):
        """Return (x0, y0, x1, y1) in figure fraction, covering all text/ticks."""
        fig_w, fig_h = fig.get_size_inches() * fig.dpi
        bbs = []
        for ax in axes_list:
            if not ax.get_visible():
                continue
            bb = ax.get_tightbbox(renderer)
            if bb is not None:
                bbs.append(bb)
        x0 = min(b.x0 for b in bbs) / fig_w
        y0 = min(b.y0 for b in bbs) / fig_h
        x1 = max(b.x1 for b in bbs) / fig_w
        y1 = max(b.y1 for b in bbs) / fig_h
        return x0, y0, x1, y1

    # Frame around the participant grid
    x0, y0, x1, y1 = _tight_bounds(part_axes)
    rect = FancyBboxPatch(
        (x0 - (PAD), y0 - PAD), (x1 - x0 + 2 * PAD), (y1 - y0 + 2 * PAD),
        boxstyle="round,pad=0.005",
        linewidth=2, edgecolor=PART_COLOR, facecolor="none",
        transform=fig.transFigure, zorder=0, clip_on=False,
    )
    fig.add_artist(rect)
    fig.text(
        x0 - (PAD), y1 + PAD + 0.01, _PART_HEADING,
        color=PART_COLOR, fontweight="bold", #fontsize=10, 
        ha="left", va="bottom", transform=fig.transFigure,
    )

    # Frame around the group-average panel
    ax0, ay0, ax1, ay1 = _tight_bounds([avg_ax])
    avg_rect = FancyBboxPatch(
        (ax0 - PAD, ay0 - PAD), (ax1 - ax0 + 2 * PAD), (ay1 - ay0 + 2 * PAD),
        boxstyle="round,pad=0.005",
        linewidth=2, edgecolor=AVG_COLOR, facecolor="none",
        transform=fig.transFigure, zorder=0, clip_on=False,
    )
    fig.add_artist(avg_rect)
    fig.text(
        ax0 - PAD, ay1 + PAD + 0.01, _AVG_HEADING,
        color=AVG_COLOR, fontweight="bold", #fontsize=10, 
        ha="left", va="bottom", transform=fig.transFigure,
    )

    figure_path = figures_dir / f"fig_fitted_vs_empirical_{cond_safe}.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    if not _in_jupyter():
        plt.close(fig)
    print(f"  Saved: {figure_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Combined figure: all conditions in one plot
# ═══════════════════════════════════════════════════════════════════════════

def _add_panel_letter(ax: plt.Axes, index: int) -> None:
    """Mark *ax* with its panel letter (A, B, C, ...) above its upper-left corner."""
    ax.text(
        0.0, 1.02, chr(ord("A") + index), transform=ax.transAxes,
        ha="left", va="bottom", fontweight="bold", fontsize=12,
    )


def _draw_compact_condition_panel(
    ax: plt.Axes,
    preds_df: pd.DataFrame,
    condition: str,
    outcome_col: str,
    *,
    show_individual: bool = True,
    individual_style: str = "fit",
    amplitude: float | None = None,
    log_bf10: float | None = None,
) -> None:
    """Draw one condition onto *ax*: group fit, ETI, block means and per-participant traces.

    Individual variability is carried by one thin line per participant rather
    than by a separate subplot grid, which is what lets all conditions share a
    single figure. With ``individual_style="fit"`` that line is the conditional
    posterior mean (fixed + random effects); with ``"observed"`` it is the
    participant's raw values, drawn as tiny connected points.
    """
    color = _condition_color(condition)

    # One thin line per participant: the spread of these lines *is* the
    # between-participant variability in level, amplitude and phase.
    if show_individual:
        for _, sub in preds_df.groupby(ANALYSIS_PARTICIPANT_COL):
            sub = sub.sort_values("hours_since_start")
            if individual_style == "observed":
                # Raw participant values instead of the smooth model fit: tiny
                # markers make the sampling times visible, the connecting line
                # keeps each participant's trajectory traceable.
                ax.plot(sub["hours_since_start"].values, sub[outcome_col].values,
                        color=color, lw=0.6, alpha=0.40, marker="o", ms=1.8,
                        markeredgewidth=0, zorder=2)
            else:
                ax.plot(sub["hours_since_start"].values, sub["mu_cond_mean"].values,
                        color=color, lw=0.8, alpha=0.45, zorder=2)

    block = _build_block_average(preds_df, outcome_col)
    x     = block["hours_since_start"].values

    # 95 % ETI of the marginal (fixed-effects) posterior, tinted with the
    # condition colour so the ribbon reads as part of that condition's panel
    ax.fill_between(
        x, block["mu_marg_lo"].values, block["mu_marg_hi"].values,
        color=color, alpha=0.10, linewidth=0, zorder=1, label="95 % ETI",
    )
    # Marginal posterior mean: the group-level cosinor fit
    ax.plot(x, block["mu_marg_mean"].values, color="black", lw=2, zorder=5,
            label="Group mean (fixed effects)")
    # Observed block means with SE, ringed in white so they stay legible on the lines
    ax.errorbar(
        x, block["obs_mean"].values, yerr=block["obs_se"].values,
        fmt="o", color=color, ms=5, capsize=2, elinewidth=1,
        markeredgecolor="white", markeredgewidth=0.8, zorder=6,
        label="Observed (block mean ± SE)",
    )

    ax.set_xlim(x.min() - 1.5, x.max() + 1.5)
    ax.set_xticks(np.arange(0, x.max() + 1, 4))
    ax.grid(True, axis="y", color="gray", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)

    # Amplitude / Bayes factor annotation in the upper-right corner
    annotation = []
    if amplitude is not None:
        annotation.append(f"Amplitude = {amplitude:.3g}")
    if log_bf10 is not None:
        # _fmt_bayes_factor() flips the sign and prints BF10/BF01 depending on the
        # direction of evidence; report log10 BF10 directly instead, consistent with
        # the tables and manuscript text.
        # # annotation.append(_fmt_bayes_factor(log_bf10))
        annotation.append(f"log$_{{10}}$ BF$_{{10}}$ = {log_bf10:.3f}")

    # The condition header sits inside the panel (upper left) rather than above
    # it, so the panels pack tighter and the header reads as part of the plot.
    # Both it and the stats annotation need clear space above the data.
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + 0.22 * (y_hi - y_lo))
    ax.text(
        0.02, 0.96, _condition_label(condition),
        transform=ax.transAxes, ha="left", va="top",
        color=readable(color), fontweight="bold",
    )
    if annotation:
        ax.text(
            0.98, 0.96, "\n".join(annotation),
            transform=ax.transAxes, ha="right", va="top",
            fontsize="small", color="black", linespacing=1.4,
        )


def _draw_condition_forest(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    conditions: list[str],
    outcome_col: str,
) -> None:
    """Draw a forest plot of the circadian amplitude estimate across conditions.

    The amplitude is non-negative by construction, so its interval never
    includes zero; the zero line is a scale reference only.  Evidence for a
    rhythm comes from the Bayes factor printed beside each row.
    """
    # Top-to-bottom reading order matches the panel order above
    y_positions = list(range(len(conditions)))[::-1]

    x_lo, x_hi = 0.0, 0.0
    for y, condition in zip(y_positions, conditions):
        est = _summary_value(summary_df, condition, "amplitude")
        lo  = _summary_value(summary_df, condition, "amplitude_ci_lo")
        hi  = _summary_value(summary_df, condition, "amplitude_ci_hi")
        if est is None:
            continue
        color = _condition_color(condition)
        if lo is not None and hi is not None:
            ax.plot([lo, hi], [y, y], color=color, lw=2.5, solid_capstyle="round", zorder=2)
            x_lo, x_hi = min(x_lo, lo), max(x_hi, hi)
        ax.plot(est, y, "o", color=color, ms=7, markeredgecolor="white",
                markeredgewidth=1.2, zorder=3)
        x_hi = max(x_hi, est)

    ax.axvline(0, color="black", linestyle=(0, (4, 4)), linewidth=0.8,
               alpha=0.5, zorder=1)

    ax.set_yticks(y_positions, [c.upper() for c in conditions])
    # Extra headroom above the top row so the in-plot header clears the bars,
    # matching how the condition panels reserve space for their own header.
    ax.set_ylim(-0.8, len(conditions) + 0.15)

    # Reserve the right-hand third of the panel for the Bayes-factor column
    span = (x_hi - x_lo) or 1.0
    ax.set_xlim(x_lo - 0.05 * span, x_hi + 0.62 * span)
    text_x = x_hi + 0.10 * span

    for y, condition in zip(y_positions, conditions):
        log_bf10 = _summary_value(summary_df, condition, "log_bf10")
        if log_bf10 is None:
            continue
        # _fmt_bayes_factor() flips the sign and prints BF10/BF01 depending on the
        # direction of evidence; report log10 BF10 directly instead, consistent with
        # the tables and manuscript text.
        # ax.text(text_x, y, _fmt_bayes_factor(log_bf10), va="center", ha="left",
        #         fontsize="small", color="black")
        ax.text(text_x, y, f"log$_{{10}}$ BF$_{{10}}$ = {log_bf10:.3f}", va="center", ha="left",
                fontsize="small", color="black")

    unit = _unit_from_label(outcome_col)
    ax.set_xlabel(f"Circadian amplitude [{unit}]" if unit else "Circadian amplitude")
    ax.text(0.02, 0.96, "Across conditions", transform=ax.transAxes,
            ha="left", va="top", fontweight="bold")
    ax.grid(True, axis="x", color="gray", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)


def plot_all_conditions(
    preds_by_condition: dict[str, pd.DataFrame],
    outcome_col: str = "amplitude",
    figures_dir: Path | None = None,
    *,
    show_stats: bool = True,
    summary_df: pd.DataFrame | None = None,
    share_y: bool = False,
    show_individual: bool = True,
    individual_style: str = "fit",
) -> Path:
    """Plot every condition in a single figure, one compact panel each.

    Replaces the five per-condition figures with one: each panel carries the
    group-level fit, its 95 % ETI, the observed block means and one thin line
    per participant, and a final forest panel compares the circadian amplitude
    and Bayes factor across conditions.

    Parameters
    ----------
    preds_by_condition : dict[str, pd.DataFrame]
        Condition key -> DataFrame loaded from that condition's *_predictions.csv.
    outcome_col : str, default 'amplitude'
        Outcome variable to plot.
    figures_dir : Path or None
        Directory to save the figure. If None, uses the default for *outcome_col*.
    show_stats : bool, default True
        Annotate each panel with the circadian amplitude and Bayes factor.
    summary_df : pd.DataFrame or None
        circadian_pupil_all_conditions.csv. Without it the annotations and the
        forest panel are skipped.
    share_y : bool, default False
        Put all condition panels on a common y-scale. Off by default because
        response magnitude differs several-fold between conditions, which would
        flatten the weaker ones.
    show_individual : bool, default True
        Draw the per-participant traces.
    individual_style : {'fit', 'observed'}, default 'fit'
        'fit' draws each participant's conditional posterior mean; 'observed'
        draws their empirical values as tiny connected points instead.

    Returns
    -------
    Path
        Path of the saved figure.
    """
    valid_styles = {"fit", "observed"}
    if individual_style not in valid_styles:
        raise ValueError(
            f"individual_style must be one of {sorted(valid_styles)}, got {individual_style!r}"
        )

    if figures_dir is None:
        figures_dir = DEFAULT_FIGURES_DIR / outcome_col
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Known conditions first, in the canonical order; anything else appended
    conditions = [c for c in CONDITION_ORDER if c in preds_by_condition]
    conditions += [c for c in preds_by_condition if c not in conditions]

    # The forest panel needs the group-level estimates from the summary table
    has_forest = (
        summary_df is not None
        and {"condition", "amplitude"}.issubset(summary_df.columns)
    )
    n_panels = len(conditions) + (1 if has_forest else 0)
    n_cols   = min(3, n_panels)
    n_rows   = int(np.ceil(n_panels / n_cols))

    fig = plt.figure(figsize=(3.9 * n_cols, 3.1 * n_rows + 0.4), constrained_layout=True)
    gs  = fig.add_gridspec(n_rows, n_cols)
    fig.suptitle(
        f"M1 fitted vs. observed {lowercase_first_word(get_parameter_label(outcome_col, title=True))} "
        "across chromatic conditions",
        fontweight="bold",
    )

    y_label = get_parameter_label(outcome_col)
    x_label = get_parameter_label("hours_since_start")

    cond_axes: list[plt.Axes] = []
    for idx, condition in enumerate(conditions):
        row, col = divmod(idx, n_cols)
        ax = fig.add_subplot(gs[row, col])
        cond_axes.append(ax)
        _draw_compact_condition_panel(
            ax, preds_by_condition[condition], condition, outcome_col,
            show_individual=show_individual,
            individual_style=individual_style,
            amplitude=_summary_value(summary_df, condition, "amplitude") if show_stats else None,
            log_bf10=_summary_value(summary_df, condition, "log_bf10") if show_stats else None,
        )
        _add_panel_letter(ax, idx)
        # Label only the outer axes to keep the grid uncluttered
        if col == 0:
            ax.set_ylabel(y_label)
        if row == n_rows - 1 or idx >= n_panels - n_cols:
            ax.set_xlabel(x_label)

    if share_y:
        lo = min(ax.get_ylim()[0] for ax in cond_axes)
        hi = max(ax.get_ylim()[1] for ax in cond_axes)
        for ax in cond_axes:
            ax.set_ylim(lo, hi)

    if has_forest:
        row, col = divmod(len(conditions), n_cols)
        forest_ax = fig.add_subplot(gs[row, col])
        _draw_condition_forest(forest_ax, summary_df, conditions, outcome_col)
        _add_panel_letter(forest_ax, len(conditions))

    # Shared legend below the grid, built from grey proxies because the panel
    # colours encode the condition rather than the layer
    handles = [
        Line2D([], [], color="black", lw=2, label="Group mean (fixed effects)"),
        Patch(facecolor=_FALLBACK_COLOR, alpha=0.20, label="95 % ETI"),
        Line2D([], [], color=_FALLBACK_COLOR, marker="o", ls="none", ms=5,
               markeredgecolor="white", markeredgewidth=0.8,
               label="Observed (block mean ± SE)"),
    ]
    if show_individual:
        if individual_style == "observed":
            handles.append(
                Line2D([], [], color=_FALLBACK_COLOR, lw=0.6, alpha=0.7,
                       marker="o", ms=1.8, markeredgewidth=0,
                       label="Participant (observed)")
            )
        else:
            handles.append(
                Line2D([], [], color=_FALLBACK_COLOR, lw=0.8, alpha=0.7,
                       label="Participant fit (fixed + random effects)")
            )
    fig.legend(handles=handles, loc="outside lower center",
               ncol=len(handles), frameon=False)

    figure_path = figures_dir / "fig_fitted_vs_empirical_all_conditions.png"
    fig.savefig(figure_path, dpi=150)
    if not _in_jupyter():
        plt.close(fig)
    print(f"  Saved: {figure_path}")
    return figure_path


def main(
    outcome_col: str = "amplitude",
    idata_dir: Path | None = None,
    figures_dir: Path | None = None,
    show_stats: bool = True,
    layout: str = "both",
    share_y: bool = False,
    individual_style: str = "observed",
) -> None:
    """
    Plot fitted-vs-empirical figures for all conditions with saved predictions.

    Reads every *_predictions.csv in the specified directory and renders either
    one detailed figure per condition, a single combined figure covering all
    conditions, or both.

    Parameters
    ----------
    outcome_col : str
        Outcome variable to plot.
    idata_dir : Path or None
        Directory containing *_predictions.csv files.
    figures_dir : Path or None
        Directory to save output figures.
    show_stats : bool
        Annotate group-average panel with amplitude and log BF10.
    layout : {'both', 'per_condition', 'combined'}
        'per_condition' renders the detailed per-participant grid per condition,
        'combined' renders all conditions in one figure, 'both' renders each.
    share_y : bool
        Put all condition panels of the combined figure on a common y-scale.
    individual_style : {'fit', 'observed'}
        How to draw per-participant traces in the combined figure: model fits
        or empirical values as tiny connected points.
    """
    valid_layouts = {"both", "per_condition", "combined"}
    if layout not in valid_layouts:
        raise ValueError(f"layout must be one of {sorted(valid_layouts)}, got {layout!r}")

    if idata_dir is None:
        idata_dir = DEFAULT_IDATA_DIR / outcome_col
    if figures_dir is None:
        figures_dir = DEFAULT_FIGURES_DIR / outcome_col

    pred_files = sorted(idata_dir.glob("*_predictions.csv"))
    if not pred_files:
        raise FileNotFoundError(
            f"No *_predictions.csv files found in {idata_dir}.\n"
            "Run analyse_circadian_pupil_parameters.py --fit first."
        )

    # Load summary table once for amplitude / BF annotations
    _results_dir = _PROJECT_DIR / "results" / "tables" / "circadian_cosinor" / outcome_col
    summary_csv  = _results_dir / "circadian_pupil_all_conditions.csv"
    summary_df   = pd.read_csv(summary_csv) if summary_csv.exists() else None
    if summary_df is None and show_stats:
        print(f"  Note: summary CSV not found at {summary_csv}; skipping amplitude/BF annotation.")

    preds_by_condition: dict[str, pd.DataFrame] = {}
    for pred_path in pred_files:
        cond_safe = pred_path.stem.replace("_predictions", "")
        data_csv  = idata_dir / f"{cond_safe}_data.csv"
        # Prefer the human-readable condition name stored in the data file over the
        # filename stem, which may differ in capitalisation or spacing
        if data_csv.exists():
            condition = str(pd.read_csv(data_csv, nrows=1)[PUPIL_CONDITION_COL].iloc[0])
        else:
            condition = cond_safe
        preds_by_condition[condition] = pd.read_csv(pred_path)

    if layout in ("per_condition", "both"):
        for condition, preds_df in preds_by_condition.items():
            print(f"\n── Condition: {condition.upper()} ──────────────────")
            plot_fitted_vs_empirical(
                preds_df, condition, outcome_col=outcome_col, figures_dir=figures_dir,
                show_stats=show_stats, summary_df=summary_df,
            )

    if layout in ("combined", "both"):
        print("\n── All conditions ──────────────────")
        plot_all_conditions(
            preds_by_condition, outcome_col=outcome_col, figures_dir=figures_dir,
            show_stats=show_stats, summary_df=summary_df, share_y=share_y,
            individual_style=individual_style,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-hoc fitted-vs-empirical plots for circadian cosinor models."
    )
    parser.add_argument(
        "--outcome",
        nargs="+",
        choices=PARAMETER_CHOICES,
        default=["amplitude", "tonic_min"],
        help="Outcome variable(s) to plot (default: amplitude tonic_min). "
             "Pass one or more values separated by spaces.",
    )
    parser.add_argument("--idata-dir", type=Path, default=None, help="Directory with *_predictions.csv files.")
    parser.add_argument("--figures-dir", type=Path, default=None, help="Directory to save output figures.")
    parser.add_argument("--no-stats", action="store_true", help="Suppress amplitude and log BF10 annotation on group-average panel.")
    parser.add_argument(
        "--layout",
        choices=["both", "per_condition", "combined"],
        default="both",
        help="'per_condition': one detailed figure per condition; 'combined': all "
             "conditions in one figure; 'both' (default): render each.",
    )
    parser.add_argument(
        "--share-y",
        action="store_true",
        help="Put all condition panels of the combined figure on a common y-scale.",
    )
    parser.add_argument(
        "--individual-style",
        choices=["fit", "observed"],
        default="observed",
        help="Per-participant traces in the combined figure: 'fit' draws "
        "conditional posterior means; 'observed' (default) draws empirical values as "
        "tiny connected points.",
    )
    args = parser.parse_args()

    for outcome in args.outcome:
        if len(args.outcome) > 1:
            print(f"\n{'='*60}\nOutcome: {outcome}\n{'='*60}")
        main(
            outcome_col=outcome,
            idata_dir=args.idata_dir,
            figures_dir=args.figures_dir,
            show_stats=not args.no_stats,
            layout=args.layout,
            share_y=args.share_y,
            individual_style=args.individual_style,
        )
