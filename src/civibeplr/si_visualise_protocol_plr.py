"""Supplement figures for the registered FLUX-normalised outcome.

Four bespoke figures plus the reused per-participant trajectory plot:

1. ``fig_plr_ratio_endpoint``       -- what maximum constriction actually measures
2. ``fig_plr_ratio_ordering``       -- per-trial vs averaged-trace amplitude
3. ``fig_plr_ratio_ratios``         -- ratio distributions by condition
4. ``fig_plr_ratio_bf_comparison``  -- Bayes factors across the three outcomes

Figures 1-3 (step 3b) go to ``results/figures/plr_ratio/``; Figure 4 and the
trajectories (step 5c) go to ``results/figures/circadian_cosinor/plr_ratio/``,
beside the primary outcomes.

Colour
------
The project condition palette was checked for colour-vision separation rather
than assumed safe.  Among the four modelled (non-FLUX) conditions it is clean:
the worst pair is mel/l-m at Delta E 10.7 under tritanopia, above the 8 floor,
with all normal-vision pairs >= 16.

FLUX is the problem: mel/flux separates by only 4.9 under protanopia and
l-m/flux by 4.3 under deuteranopia, and mel/flux sits at 11.3 in normal vision,
below the 15 floor.  So wherever FLUX appears beside the others it carries
secondary encoding -- a dashed line and a "(denominator)" legend label -- and identity never
rests on hue alone.  That doubles as meaning here, since FLUX is the
denominator rather than one condition among five.

Figure 4 puts three outcomes on one axis.  That is legitimate only because
log10 BF10 is unitless; the amplitudes themselves are in different units and
are never plotted on a shared scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from civibeplr.analysis_io import PUPIL_BLOCK_COL, PUPIL_CONDITION_COL, PUPIL_PARTICIPANT_COL
from civibeplr.si_derive_protocol_plr import (
    BASELINE_START_SEC,
    baseline_mask,
    cast_integral_columns,
    DEFAULT_DERIV_DIR,
    DEFAULT_FINAL_DIR,
    FLUX_CONDITION,
    NON_FLUX_CONDITIONS,
    TIME_COL,
    TRIAL_KEY,
    select_pupil_size,
)
from civibeplr.latex_table_creation import format_latex_variables
from civibeplr.load_final_data import _iter_csvs
from civibeplr.vis_io import (
    TRIAL_TYPE_COLORS,
    apply_font_style,
    readable,
    save_figure,
)

apply_font_style()

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FIG_DIR = DEFAULT_PROJECT_DIR / "results" / "figures" / "plr_ratio"
DEFAULT_CIRCADIAN_FIG_DIR = DEFAULT_PROJECT_DIR / "results" / "figures" / "circadian_cosinor"
DEFAULT_TABLES_DIR = DEFAULT_PROJECT_DIR / "results" / "tables"
DEFAULT_TEX_DIR = DEFAULT_PROJECT_DIR / "results" / "tex" / "plr_ratio"

STIM_START, STIM_END = 0.0, 5.0

#: Stimulus modulation frequency and the troughs of the pupil response to it.
#: FIRST_TROUGH_SEC is read off the first trough of the grand-mean trace
#: (~1.70 s; 1.65-1.80 s in the oscillating conditions) and rounded; the
#: second is one stimulus period later.  These are pupil-response troughs, not
#: the stimulus contrast troughs (1.875 and 4.375 s), and because they come
#: from the same trials the share of minima near them is descriptive.  The pupil is
#: still completing its onset constriction before the first trough, so
#: oscillation depth is measured over exactly one cycle between successive
#: troughs; starting earlier measures the onset ramp instead of the modulation.
STIM_FREQ_HZ = 0.4
STIM_PERIOD_SEC = 1.0 / STIM_FREQ_HZ
FIRST_TROUGH_SEC = 1.75
SECOND_TROUGH_SEC = FIRST_TROUGH_SEC + STIM_PERIOD_SEC

#: Order used in every panel: the four modelled conditions, then FLUX last
#: because it is the denominator rather than a fifth series.
PLOT_ORDER = ["mel", "s", "l-m", "lms", FLUX_CONDITION]

#: Styling shared with the main-pipeline figures (visualise_circadian_parameter_cosinor):
#: boxed axes, a faint grey grid, black ink, panel letters above each axis and
#: upper-case condition labels.  Take-home messages go in the appendix
#: captions, not into the figures.
GRID_KW = dict(color="gray", linestyle="-", linewidth=0.5, alpha=0.25)
#: Neutral grey for legend proxies whose colour would otherwise suggest a condition.
NEUTRAL = "#787878"
#: Pupil-response troughs in the endpoint figure; distinct from the grey stimulus shading.
TROUGH_COLOR = "#C9A227"

#: Outcomes compared in the Bayes-factor figure: (metric, legend label, marker).
BF_OUTCOMES = [
    ("amplitude", "Phasic amplitude", "o"),
    ("tonic_min", "Tonic minimum", "s"),
    ("plr_ratio", "Max. constriction / FLUX (registered)", "D"),
]
#: SD/|dlogZ| above which a Bayes factor is flagged as numerically imprecise;
#: the pragmatic threshold stated in the Methods.
BF_PRECISION_THRESHOLD = 0.10


def _add_panel_letter(ax: plt.Axes, index: int) -> None:
    """Panel letter above the upper-left corner, as in the main-pipeline figures."""
    ax.text(
        0.0, 1.02, chr(ord("A") + index), transform=ax.transAxes,
        ha="left", va="bottom", fontweight="bold", fontsize=12,
    )


def _grid(ax: plt.Axes, axis: str = "y") -> None:
    ax.grid(True, axis=axis, **GRID_KW)
    ax.set_axisbelow(True)


# ═══════════════════════════════════════════════════════════════════════════
# Trace data (recomputed from the cleaned files; cheap enough not to cache)
# ═══════════════════════════════════════════════════════════════════════════

_SAMPLE_KEY = [PUPIL_PARTICIPANT_COL, PUPIL_CONDITION_COL, PUPIL_BLOCK_COL]


def compute_grand_mean_traces(
    final_dir: Path | None = None,
    bin_hz: float = 20.0,
    sample_blocks: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (grand-mean baseline-corrected traces, per-trial minimum times).

    If *sample_blocks* is given, only its participant-condition-block
    combinations are used (the analysis sample).
    """
    final_dir = final_dir or DEFAULT_FINAL_DIR
    trace_frames, min_frames = [], []
    keep = None
    if sample_blocks is not None:
        keep = pd.MultiIndex.from_frame(sample_blocks[_SAMPLE_KEY])

    for path in _iter_csvs(final_dir, "[0-9][0-9][0-9]_*.csv"):
        df = pd.read_csv(path, index_col=0)
        if keep is not None:
            df = df[pd.MultiIndex.from_frame(df[_SAMPLE_KEY]).isin(keep)]
            if df.empty:
                continue
        df["pupil_size_mm"] = select_pupil_size(df)
        t = df[TIME_COL]

        base = (
            df[baseline_mask(df)]
            .groupby(TRIAL_KEY, dropna=False)["pupil_size_mm"]
            .mean()
            .rename("_baseline")
        )
        df = df.join(base, on=TRIAL_KEY)
        df["_rel"] = df["pupil_size_mm"] - df["_baseline"]

        stim = df[(t >= STIM_START) & (t <= STIM_END)].dropna(subset=["_rel"])
        if stim.empty:
            continue

        # Participant-level mean trace, so every participant weighs equally.
        binned = df[(t >= BASELINE_START_SEC) & (t <= STIM_END + 1.0)].copy()
        binned["_tb"] = (binned[TIME_COL] * bin_hz).round() / bin_hz
        trace_frames.append(
            binned.groupby([PUPIL_CONDITION_COL, "_tb"])["_rel"].mean()
        )

        idx = stim.groupby(TRIAL_KEY, dropna=False)["_rel"].idxmin()
        min_frames.append(stim.loc[idx, [PUPIL_CONDITION_COL, TIME_COL]])

    traces = pd.concat(trace_frames, axis=1).mean(axis=1).unstack(PUPIL_CONDITION_COL)
    min_times = pd.concat(min_frames, ignore_index=True).rename(
        columns={TIME_COL: "min_time_s"}
    )
    return traces, min_times


def export_trace_variables(
    traces: pd.DataFrame, min_times: pd.DataFrame, tex_dir: Path | None = None
) -> Path:
    """Export trace-level quantities the appendix quotes, as LaTeX macros.

    Oscillation depth is the peak-to-trough range of the *grand-mean* trace
    over one stimulus cycle, between successive troughs (1.75-4.25 s).  It has to be measured on the grand mean:
    per block, peak-to-trough is dominated by measurement noise and makes L-M --
    the least modulated condition -- look like the most modulated one.
    """
    tex_dir = tex_dir or DEFAULT_TEX_DIR
    tex_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for cond in traces.columns:
        sub = min_times[min_times[PUPIL_CONDITION_COL] == cond]["min_time_s"]
        near = ((sub - FIRST_TROUGH_SEC).abs() < 0.5) | ((sub - SECOND_TROUGH_SEC).abs() < 0.5)
        rows.append({
            "condition": cond,
            "min_time_median_s": float(sub.median()) if len(sub) else np.nan,
            "min_near_trough_pct": float(100 * near.mean()) if len(sub) else np.nan,
        })
    df = pd.DataFrame(rows)
    all_min = min_times["min_time_s"]
    near_all = ((all_min - FIRST_TROUGH_SEC).abs() < 0.5) | ((all_min - SECOND_TROUGH_SEC).abs() < 0.5)
    df = pd.concat([df, pd.DataFrame([{
        "condition": "all",
        "min_near_trough_pct": float(100 * near_all.mean()),
        "n_trial_minima": int(len(all_min)),
    }])], ignore_index=True)

    # Whole percent, matching the share printed in the endpoint figure legend.
    df["min_near_trough_pct"] = df["min_near_trough_pct"].round(0)
    df = cast_integral_columns(df, key_col="condition")
    out = tex_dir / "vars_plr_ratio_traces.tex"
    out.write_text(
        format_latex_variables(df, row_key_col="condition", fmt=".3f",
                               prefix="plr_ratio_trace"),
        encoding="utf-8")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 -- what the endpoint measures
# ═══════════════════════════════════════════════════════════════════════════

def _participant_summary(
    block_df: pd.DataFrame, value_col: str, order: list[str]
) -> tuple[list[float], list[float], list[pd.Series]]:
    """Mean and SEM across participant means, per condition in *order*.

    Blocks are repeated observations within participants, so they are
    averaged within participant first; the SEM then reflects between-
    participant uncertainty (n = participants) rather than the number of
    blocks.  Also returns each condition's participant means for plotting.
    """
    means, sems, pp_means = [], [], []
    for cond in order:
        sub = block_df[block_df[PUPIL_CONDITION_COL] == cond].dropna(subset=[value_col])
        pp = sub.groupby(PUPIL_PARTICIPANT_COL)[value_col].mean()
        means.append(float(pp.mean()))
        sems.append(float(pp.sem()))
        pp_means.append(pp)
    return means, sems, pp_means


def plot_endpoint_characterisation(
    traces: pd.DataFrame,
    min_times: pd.DataFrame,
    block_df: pd.DataFrame,
    figures_dir: Path | None = None,
) -> Path:
    """The diagnosis figure: maximum constriction is tonic-dominated.

    Panel A -- grand-mean baseline-corrected traces.  L-M constricts deepest
    while oscillating least, which is why it scores highest on an endpoint that
    reads the single deepest sample.

    Panel B -- when the minimum occurs, one row per condition.  Clustering at
    the pupil-response troughs shows the endpoint does track the stimulus; how tightly
    differs by condition.

    Panel C -- mean block amplitude by condition, with the manuscript's phasic
    ordering for contrast.
    """
    figures_dir = figures_dir or DEFAULT_FIG_DIR
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14.0, 4.4))

    # ── A: grand-mean traces ──────────────────────────────────────────────
    # Restricted to the endpoint window: the registered maximum is searched
    # over 0-5 s, so post-stimulus recovery is not part of the argument.
    window = traces.loc[BASELINE_START_SEC:STIM_END]
    for cond in PLOT_ORDER:
        if cond not in window.columns:
            continue
        is_flux = cond == FLUX_CONDITION
        ax_a.plot(
            window.index, window[cond],
            color=TRIAL_TYPE_COLORS[cond], lw=2.0,
            ls="--" if is_flux else "-", zorder=3,
            label=cond.upper() + (" (denominator)" if is_flux else ""),
        )
    # Same stimulus shading as the main-article trace figure.
    ax_a.axvspan(STIM_START, STIM_END, color="grey", alpha=0.2, zorder=0, linewidth=0)
    ax_a.axhline(0, color="gray", lw=0.8, zorder=1)
    ax_a.set_xlim(BASELINE_START_SEC, STIM_END)
    ax_a.set_xlabel("Trial time [s]")
    ax_a.set_ylabel("Rel. pupil size [mm]")
    ax_a.set_title("Grand-mean response")
    ax_a.legend(frameon=False, loc="lower left", fontsize="small")
    _grid(ax_a)
    _add_panel_letter(ax_a, 0)

    # ── B: when the minimum falls ─────────────────────────────────────────
    # One row per condition on a shared time axis: overlaid histograms of five
    # conditions mix into unreadable colours.  The +/-0.5 s trough windows are
    # shaded so each row's near-trough share refers to a visible region.
    troughs = [FIRST_TROUGH_SEC, SECOND_TROUGH_SEC]
    bins = np.arange(0, 5.125, 0.125)
    b_order = [c for c in PLOT_ORDER
               if c in set(min_times[PUPIL_CONDITION_COL])]
    b_grid = ax_b.get_subplotspec().subgridspec(len(b_order), 1, hspace=0.15)
    ax_b.remove()
    b_axes = []
    for k, cond in enumerate(b_order):
        ax = fig.add_subplot(b_grid[k], sharex=b_axes[0] if b_axes else None)
        b_axes.append(ax)
        t_min = min_times.loc[min_times[PUPIL_CONDITION_COL] == cond, "min_time_s"]
        near = np.zeros(len(t_min), dtype=bool)
        for t_trough in troughs:
            near |= ((t_min - t_trough).abs() < 0.5).to_numpy()
            ax.axvspan(t_trough - 0.5, t_trough + 0.5, color=TROUGH_COLOR,
                       alpha=0.15, zorder=0, linewidth=0)
            ax.axvline(t_trough, color=TROUGH_COLOR, lw=1.0, ls=":", zorder=4)
        ax.hist(t_min, bins=bins, color=TRIAL_TYPE_COLORS[cond],
                edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.text(0.01, 0.92, cond.upper(), transform=ax.transAxes,
                ha="left", va="top", fontsize="small",
                color=readable(TRIAL_TYPE_COLORS[cond]))
        ax.text(0.99, 0.92, f"{near.mean():.0%}", transform=ax.transAxes,
                ha="right", va="top", fontsize="small")
        ax.tick_params(axis="y", labelsize="x-small")
        if k < len(b_order) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        _grid(ax)
    b_axes[0].set_xlim(0, STIM_END)
    b_axes[0].set_title("Timing of the maximum constriction", pad=16)
    b_axes[0].legend(
        handles=[
            Line2D([], [], color=TROUGH_COLOR, lw=1.2, ls=":",
                   label="Pupil-response trough"),
            Patch(facecolor=TROUGH_COLOR, alpha=0.15,
                  label="±0.5 s window (% of trial minima inside)"),
        ],
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.98),
        ncol=2, fontsize="x-small", handlelength=1.5, columnspacing=1.0,
    )
    b_axes[-1].set_xlabel("Time of trial minimum after stimulus onset [s]")
    b_axes[len(b_axes) // 2].set_ylabel("Number of trials")
    _add_panel_letter(b_axes[0], 1)

    # ── C: mean amplitude by condition (participant level) ────────────────
    order = [c for c in PLOT_ORDER if c in set(block_df[PUPIL_CONDITION_COL])]
    means, sems, pp_means = _participant_summary(block_df, "mean_constriction_mm", order)
    x = np.arange(len(order))
    ax_c.bar(x, means, width=0.6, color=[TRIAL_TYPE_COLORS[c] for c in order],
             alpha=0.85, zorder=3)
    for xi, pp in zip(x, pp_means):
        ax_c.scatter(np.full(len(pp), xi), pp.values, s=9, color="black",
                     alpha=0.45, edgecolors="none", zorder=4)
    ax_c.errorbar(x, means, yerr=sems, fmt="none", ecolor="black",
                  elinewidth=1.0, capsize=3, zorder=5)
    top = [max(m + e, pp.max()) for m, e, pp in zip(means, sems, pp_means)]
    for xi, m, t in zip(x, means, top):
        ax_c.annotate(f"{m:.2f}", xy=(xi, t), xytext=(0, 4),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize="small")
    ax_c.set_ylim(0, max(top) * 1.15)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([c.upper() for c in order])
    ax_c.set_ylabel("Mean block amplitude [mm]")
    ax_c.set_title("Registered endpoint by condition")
    _grid(ax_c)
    _add_panel_letter(ax_c, 2)

    fig.suptitle("What the registered maximum constriction measures")
    fig.tight_layout()
    out = figures_dir / "fig_plr_ratio_endpoint.png"
    save_figure(fig, out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 -- order of operations
# ═══════════════════════════════════════════════════════════════════════════

def plot_ordering_audit(block_df: pd.DataFrame, figures_dir: Path | None = None) -> Path:
    """Per-trial maxima vs the maximum of the trial-averaged trace."""
    figures_dir = figures_dir or DEFAULT_FIG_DIR
    d = block_df.dropna(subset=["mean_constriction_mm", "mean_constriction_avgtrace_mm"])
    order = [c for c in PLOT_ORDER if c in set(d[PUPIL_CONDITION_COL])]
    fig = plt.figure(figsize=(9.0, 4.4))
    grid = fig.add_gridspec(
        2, len(order), height_ratios=[1, 0.85],
        left=0.085, right=0.985, bottom=0.11, top=0.88,
        wspace=0.12, hspace=0.50,
    )
    scatter_axes = []

    lim = (
        min(d["mean_constriction_avgtrace_mm"].min(), d["mean_constriction_mm"].min()) - 0.1,
        max(d["mean_constriction_avgtrace_mm"].max(), d["mean_constriction_mm"].max()) + 0.1,
    )
    for i, cond in enumerate(order):
        ax_a = fig.add_subplot(grid[0, i])
        scatter_axes.append(ax_a)
        ax_a.plot(lim, lim, color="gray", lw=1.0, ls="--", zorder=2)
        sub = d[d[PUPIL_CONDITION_COL] == cond]
        ax_a.scatter(
            sub["mean_constriction_avgtrace_mm"], sub["mean_constriction_mm"],
            s=10, color=TRIAL_TYPE_COLORS[cond], alpha=0.65,
            edgecolors="none", zorder=3,
        )
        ax_a.set_xlim(*lim); ax_a.set_ylim(*lim)
        ax_a.set_aspect("equal", adjustable="box")
        ax_a.set_xticks([0, 1, 2, 3]); ax_a.set_yticks([0, 1, 2, 3])
        ax_a.tick_params(labelsize=8, labelleft=i == 0)
        ax_a.set_title(cond.upper(), fontsize=10, fontweight="bold",
                       color=readable(TRIAL_TYPE_COLORS[cond]))
        _grid(ax_a, axis="both")
    scatter_axes[0].set_ylabel("Mean per-trial\nmaximum [mm]", fontsize=9)
    _add_panel_letter(scatter_axes[0], 0)
    fig.text(0.535, 0.965, "Amplitude comparison", ha="center", fontsize=11)
    fig.text(0.535, scatter_axes[0].get_position().y0 - 0.075,
             "Maximum of averaged trace [mm]", ha="center", fontsize=9)

    ax_b = fig.add_subplot(grid[1, :])
    bias, sem, pp_bias = _participant_summary(d, "ordering_bias_mm", order)
    x = np.arange(len(order))
    for xi, b, e, pp, cond in zip(x, bias, sem, pp_bias, order):
        ax_b.scatter(np.full(len(pp), xi + 0.12), pp.values, s=9, color="black",
                     alpha=0.45, edgecolors="none", zorder=3)
        ax_b.errorbar(xi, b, yerr=e, fmt="o", color=TRIAL_TYPE_COLORS[cond],
                      markersize=6, markeredgewidth=0, elinewidth=1.5,
                      capsize=0, zorder=4)
    # Participant means can fall below zero, so pad both ends of the range
    top = max(max(np.add(bias, sem)), max(pp.max() for pp in pp_bias))
    bottom = min(0.0, min(pp.min() for pp in pp_bias))
    pad = 0.12 * (top - bottom)
    ax_b.set_ylim(bottom - (pad if bottom < 0 else 0.0), top + pad)
    ax_b.set_xlim(-0.5, len(order) - 0.5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([c.upper() for c in order])
    ax_b.set_ylabel("Per-trial - averaged-trace\namplitude [mm]", fontsize=9)
    ax_b.tick_params(labelsize=9)
    ax_b.set_title("Mean difference", fontsize=11)
    _grid(ax_b)
    _add_panel_letter(ax_b, 1)

    out = figures_dir / "fig_plr_ratio_ordering.png"
    save_figure(fig, out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 -- ratio distributions
# ═══════════════════════════════════════════════════════════════════════════

def plot_ratio_distributions(
    analysis_df: pd.DataFrame, figures_dir: Path | None = None
) -> Path:
    """Ratio by condition: observation spread and participant means."""
    figures_dir = figures_dir or DEFAULT_FIG_DIR
    fig, ax = plt.subplots(figsize=(6.6, 3.8))

    present = set(analysis_df[PUPIL_CONDITION_COL])
    order = [c for c in PLOT_ORDER if c in NON_FLUX_CONDITIONS and c in present]
    rng = np.random.default_rng(1072)

    ax.axhline(1.0, color="gray", lw=1.0, ls="--", zorder=1)
    for i, cond in enumerate(order):
        sub = analysis_df[analysis_df[PUPIL_CONDITION_COL] == cond]
        vals = sub["plr_ratio"].dropna().values
        colour = TRIAL_TYPE_COLORS[cond]

        # The right-hand density and left-hand observations follow the
        # manuscript's parameter-distribution layout. Keep the density and
        # median block-level; participant means remain a separate layer.
        centre = i + 0.04
        parts = ax.violinplot([vals], positions=[centre], widths=0.72,
                              showextrema=False, showmedians=False)
        for body in parts["bodies"]:
            for path in body.get_paths():
                path.vertices[:, 0] = np.maximum(path.vertices[:, 0], centre)
            body.set_facecolor(colour); body.set_alpha(0.35)
            body.set_edgecolor(colour); body.set_linewidth(0.8)
            body.set_zorder(2)

        ax.scatter(i + rng.uniform(-0.29, -0.13, len(vals)), vals,
                   s=10, color=colour, alpha=0.40, linewidth=0, zorder=3)

        pmeans = sub.groupby(PUPIL_PARTICIPANT_COL)["plr_ratio"].mean()
        ax.scatter(i + rng.uniform(-0.065, -0.015, len(pmeans)), pmeans.values,
                   s=28, marker="D", color=colour, edgecolors="none", zorder=5)

        med = float(np.median(vals))
        ax.plot([centre, centre + 0.17], [med, med], color="black", lw=1.5, zorder=6)

    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([c.upper() for c in order])
    ax.set_ylabel("Maximum constriction / FLUX (R)")
    ax.set_title("FLUX-normalised maximum constriction", fontsize=11)
    _grid(ax)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", color=NEUTRAL, markersize=3.5,
                   alpha=0.6, markeredgewidth=0, label="Participant-block"),
            Line2D([], [], marker="D", ls="none", color=NEUTRAL, markersize=5,
                   markeredgewidth=0, label="Participant mean"),
            Line2D([], [], color="black", lw=1.5, label="Block median"),
            Line2D([], [], color="gray", lw=1.0, ls="--", label="R = 1"),
        ],
        frameon=False, fontsize=8, loc="lower center", ncols=4,
        handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(0.5, 0),
    )
    fig.tight_layout(pad=0.6, rect=(0, 0.075, 1, 1))
    out = figures_dir / "fig_plr_ratio_ratios.png"
    save_figure(fig, out)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 -- Bayes factors across outcomes
# ═══════════════════════════════════════════════════════════════════════════

def plot_bf_comparison(
    bf_df: pd.DataFrame, figures_dir: Path | None = None
) -> Path:
    """log10 BF10 for each condition under each outcome.

    Laid out like the "Across conditions" forest panel of the main figure:
    one row per condition, log10 BF10 on the x axis.  The thin line spans the
    three outcomes so the spread across outcomes reads as one mark.  Whiskers
    are +/- 1 SD of the log Z difference across SMC chains, and open markers
    flag estimates above the SD/|dlogZ| precision threshold used in the
    Methods.  Only the sign is marked as a reference: the Methods define no
    evidence categories, so none are drawn.

    Unitless, so a shared axis is legitimate here.  The underlying amplitudes
    are in different units and are never plotted on a common scale.
    """
    figures_dir = figures_dir or DEFAULT_FIG_DIR
    fig, ax = plt.subplots(figsize=(6.6, 3.0))

    order = [c for c in PLOT_ORDER if c in NON_FLUX_CONDITIONS]
    # Top-to-bottom reading order, as in the main forest panel
    y_positions = list(range(len(order)))[::-1]
    markers = {metric: marker for metric, _, marker in BF_OUTCOMES}

    ax.axvline(0, color="gray", lw=1.0, zorder=1)
    for y, cond in zip(y_positions, order):
        sub = bf_df[bf_df["condition"] == cond].dropna(subset=["log_bf10"])
        if sub.empty:
            continue
        colour = TRIAL_TYPE_COLORS[cond]
        ax.plot([sub["log_bf10"].min(), sub["log_bf10"].max()], [y, y],
                color=colour, lw=2.5, alpha=0.35, solid_capstyle="round", zorder=2)
        for _, row in sub.iterrows():
            x = row["log_bf10"]
            if pd.notna(row["log_bf10_sd"]):
                ax.plot([x - row["log_bf10_sd"], x + row["log_bf10_sd"]], [y, y],
                        color="black", lw=0.9, zorder=3)
            imprecise = row["sd_over_difference"] > BF_PRECISION_THRESHOLD
            ax.plot(x, y, markers[row["metric"]], ms=7, zorder=4,
                    color=colour, markerfacecolor="white" if imprecise else colour,
                    markeredgecolor=colour if imprecise else "white",
                    markeredgewidth=1.4 if imprecise else 1.0)

    ax.set_yticks(y_positions, [c.upper() for c in order])
    ax.set_ylim(-0.6, len(order) - 0.2)
    lo = min(float(bf_df["log_bf10"].min()), -1.0)
    hi = max(float(bf_df["log_bf10"].max()), 1.0)
    pad = 0.08 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("log$_{10}$ BF$_{10}$")
    for x, ha, txt in [(-0.08, "right", "favours M0"), (0.08, "left", "favours M1")]:
        ax.text(x, len(order) - 0.35, txt, ha=ha, va="center",
                fontsize="small", color=NEUTRAL)
    ax.set_title("Evidence for circadian modulation by outcome", fontsize=11)
    _grid(ax, axis="x")

    fig.legend(
        handles=[
            Line2D([], [], marker=m, ls="none", color=NEUTRAL, markersize=6,
                   markeredgecolor="white", label=lbl)
            for _, lbl, m in BF_OUTCOMES
        ] + [
            Line2D([], [], marker="o", ls="none", color=NEUTRAL, markersize=6,
                   markerfacecolor="white", markeredgewidth=1.4,
                   label=f"SD/|$\\Delta$log Z| > {BF_PRECISION_THRESHOLD:.2f}"),
            Line2D([], [], color="black", lw=0.9, label="$\\pm$1 SD across SMC chains"),
        ],
        frameon=False, fontsize=8, loc="lower center", ncols=3,
        handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(0.5, 0),
    )
    fig.tight_layout(pad=0.6, rect=(0, 0.13, 1, 1))
    out = figures_dir / "fig_plr_ratio_bf_comparison.png"
    save_figure(fig, out)
    return out



# ═══════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def collect_bf_table(outcome_col: str = "plr_ratio") -> pd.DataFrame:
    """Gather log10 BF10 and its SMC precision for the three outcomes.

    The precision columns come from the step 5b diagnostics; they are left NaN
    when that step has not been run, and the figure then omits the whiskers.
    """
    rows = []
    for metric, _, _ in BF_OUTCOMES:
        source = outcome_col if metric == "plr_ratio" else metric
        p = (DEFAULT_TABLES_DIR / "circadian_cosinor" / source
             / "circadian_pupil_all_conditions.csv")
        if not p.exists():
            continue
        d = pd.read_csv(p)[["condition", "log_bf10"]]
        d["metric"] = metric
        diag = (DEFAULT_TABLES_DIR / "circadian_cosinor_diagnostics" / source
                / "diagnostics_summary.csv")
        if diag.exists():
            g = pd.read_csv(diag)
            # Pooled SD of the per-chain log Z difference, in log10 units
            g["log_bf10_sd"] = np.hypot(g["logZ_M0_SD"], g["logZ_M1_SD"]) / np.log(10)
            g = g.rename(columns={"bf_SD_over_difference": "sd_over_difference"})
            d = d.merge(g[["condition", "log_bf10_sd", "sd_over_difference"]],
                        on="condition", how="left")
        else:
            d["log_bf10_sd"] = np.nan
            d["sd_over_difference"] = np.nan
        rows.append(d)
    if not rows:
        raise FileNotFoundError("No result CSVs found; run the model step first.")
    return pd.concat(rows, ignore_index=True)


def _print_written(written: list[Path]) -> None:
    print("\nFigures written (paths relative to the project root):")
    for path in written:
        try:
            rel = path.resolve().relative_to(DEFAULT_PROJECT_DIR)
        except ValueError:
            rel = path
        print(f"  {rel}")


def endpoint_figures(
    figures_dir: Path | None = None,
    deriv_dir: Path | None = None,
) -> list[Path]:
    """Figures 1-3: what the endpoint measures, ordering audit, ratio spread.

    Supplementary pipeline step 3b.  Needs only the derivation outputs, not
    the circadian models.
    """
    figures_dir = figures_dir or DEFAULT_FIG_DIR
    deriv_dir = deriv_dir or DEFAULT_DERIV_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)

    blocks = pd.read_csv(deriv_dir / "block_amplitudes.csv")
    analysis = pd.read_csv(deriv_dir / "paired_ratios_analysis.csv")
    # Descriptives use the analysis sample, matching the main-article means
    blocks = blocks[blocks["in_analysis_sample"]]

    print("Computing grand-mean traces ...")
    traces, min_times = compute_grand_mean_traces(sample_blocks=blocks)

    export_trace_variables(traces, min_times)

    written = [
        plot_endpoint_characterisation(traces, min_times, blocks, figures_dir),
        plot_ordering_audit(blocks, figures_dir),
        plot_ratio_distributions(analysis, figures_dir),
    ]
    _print_written(written)
    return written


def circadian_figures(
    outcome_col: str = "plr_ratio",
    trajectories: bool = True,
) -> list[Path]:
    """Figures 4-5: Bayes factors across outcomes and fitted trajectories.

    Supplementary pipeline step 5c, run per outcome.  The trajectories are the
    primary pipeline's step 5c, called unchanged: the outputs sit in the same
    ``circadian_cosinor/<outcome>/`` layout, so it finds them by outcome name.
    Figure 4 compares the registered outcome with the manuscript metrics, so it
    is only drawn for ``plr_ratio``.
    """
    figures_dir = DEFAULT_CIRCADIAN_FIG_DIR / outcome_col
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if outcome_col == "plr_ratio":
        written.append(plot_bf_comparison(collect_bf_table(outcome_col), figures_dir))

    if trajectories:
        from civibeplr.visualise_circadian_parameter_cosinor import (
            main as visualise_circadian,
        )

        visualise_circadian(outcome_col=outcome_col)
        written.extend(sorted(figures_dir.glob("fig_fitted_vs_empirical_*.png")))

    _print_written(written)
    return written


def main(
    figures_dir: Path | None = None,
    deriv_dir: Path | None = None,
    outcome_col: str = "plr_ratio",
    trajectories: bool = True,
) -> list[Path]:
    """Build every supplement figure for the registered outcome.

    Equivalent to running supplementary pipeline steps 3b and 5c for one
    outcome.  Returns the paths written.  Inside a Jupyter kernel the figures
    also render inline: ``save_figure`` leaves the figure open when
    ``_in_jupyter()`` is true, and ``%matplotlib inline`` then draws whatever
    is still open at the end of the cell.  That is the same mechanism
    ``pipeline.ipynb`` relies on, so no explicit ``display()`` call is needed.
    """
    return endpoint_figures(figures_dir, deriv_dir) + circadian_figures(
        outcome_col, trajectories
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Supplement figures for the protocol PLR outcome.")
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--outcome", default="plr_ratio")
    parser.add_argument("--no-trajectories", action="store_true")
    args = parser.parse_args()
    main(
        figures_dir=args.figures_dir,
        outcome_col=args.outcome,
        trajectories=not args.no_trajectories,
    )


if __name__ == "__main__":
    _cli()
