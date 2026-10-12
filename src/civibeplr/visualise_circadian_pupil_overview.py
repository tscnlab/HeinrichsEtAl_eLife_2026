"""
Overview figure for circadian pupil (0.4 Hz modulation amplitude) results.

NOT part of the standard analysis pipeline. This script is an exploratory /
inspection helper: it is not called by ``pipeline.py``, and nothing it produces
is copied into the manuscript or the reported figures/tables. The figure written
under ``results/figures/circadian_cosinor/`` is for local viewing only.

Three panels per figure:
  A  Histogram of amplitude across all participants (per-participant median lines)
  B  Polar plot of circadian amplitude (mm) vs. phase per participant
  C  Example trajectories for two participants with posterior cosinor fit

Usage
-----
    python -m civibeplr.visualise_circadian_pupil_overview [--condition flux]
                                                           [--pid1 211] [--pid2 213]

Outputs
-------
    results/figures/circadian_cosinor/<condition>/fig_circadian_overview_mm.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy

from civibeplr.vis_io import CONDITION_LABELS, apply_font_style

apply_font_style()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DERIV_DIR = PROJECT_ROOT / "data" / "derivatives" / "circadian_cosinor" / "amplitude"
FIG_DIR = PROJECT_ROOT / "results" / "figures" / "circadian_cosinor"

BLOCK_INTERVAL_HOURS = 3.75


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_condition_data(condition: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = DERIV_DIR / condition
    data = pd.read_csv(base.with_name(f"{condition}_data.csv"))
    blups = pd.read_csv(base.with_name(f"{condition}_blups.csv"))
    predictions = pd.read_csv(base.with_name(f"{condition}_predictions.csv"))
    return data, blups, predictions


# ---------------------------------------------------------------------------
# Panel A: histogram of outcome across all participants
# ---------------------------------------------------------------------------

def plot_outcome_histogram(
    fig: plt.Figure,
    ax: plt.Axes,
    data: pd.DataFrame,
    y_col: str,
    xlabel: str,
    panel_label: str = "A",
) -> plt.Axes:
    data = data.dropna(subset=[y_col]).copy()

    ax.hist(
        data[y_col],
        bins=25,
        facecolor="#7BC8F650",
        edgecolor="black",
        linewidth=0.8,
    )

    unique_pids = sorted(data["participant_id"].unique())
    for i, pid in enumerate(unique_pids):
        med = data.loc[data["participant_id"] == pid, y_col].median()
        ax.axvline(
            x=med,
            linestyle="-",
            linewidth=2.5,
            color=plt.cm.tab20(i),
            alpha=0.6,
            label=str(pid),
        )

    overall_med = data[y_col].median()
    ax.axvline(
        x=overall_med,
        color="k",
        linestyle=":",
        linewidth=2,
        label=f"Overall median:\n{overall_med:.3f}",
    )

    ax.grid(True, which="major", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.set_ylim(-5, ax.get_ylim()[1] * 1.25)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title("Global distribution and median levels", pad=36, loc="left")

    legend = ax.legend(
        loc="upper right",
        title="Participants",
        ncol=2,
        frameon=True,
        facecolor="#FFEEB330",
        edgecolor="#FFEEB330",
        fontsize="small",
    )
    legend.get_frame().set_alpha(0.3)

    offset = offset_copy(ax.transAxes, fig=fig, x=-2, units="points")
    ax.text(
        -0.11, 1.18, s=panel_label,
        transform=offset, fontsize=12, fontweight="bold", va="center", ha="right",
    )
    return ax


# ---------------------------------------------------------------------------
# Panel B: polar plot of amplitude (mm) vs. phase per participant
# ---------------------------------------------------------------------------

def plot_polar_circadian(
    fig: plt.Figure,
    ax: plt.Axes,
    blups: pd.DataFrame,
    panel_label: str = "B",
) -> plt.Axes:
    blups = blups.copy()
    # Phase stored as atan2 result; wrap to [0, 2*pi] for polar display
    blups["phase_wrap"] = np.mod(blups["phase_mean"], 2 * np.pi)

    unique_pids = sorted(blups["participant_id"].unique())
    pid_colours = {
        pid: plt.cm.tab20.colors[i] for i, pid in enumerate(unique_pids)
    }

    ymax = blups["amplitude_mean"].max() * 1.25

    for _, row in blups.iterrows():
        pid = row["participant_id"]
        colour = pid_colours[pid]
        ax.plot(
            [0, row["phase_wrap"]],
            [0, ymax],
            color=colour,
            linewidth=2,
            alpha=0.8,
        )
        ax.scatter(
            row["phase_wrap"],
            row["amplitude_mean"],
            s=45,
            facecolor=(*colour[:3], 0.5),
            edgecolor=colour,
            linewidth=1.5,
            zorder=3,
        )
        # Error bar (95% ETI) as thin arc -- approximated with radial error line
        ax.plot(
            [row["phase_wrap"], row["phase_wrap"]],
            [row["amplitude_lo"], row["amplitude_hi"]],
            color=colour,
            linewidth=1.2,
            alpha=0.5,
        )

    handles = [
        plt.Line2D(
            [0], [0],
            marker="o",
            color=(*c[:3], 0.5),
            markerfacecolor=(*c[:3], 0.5),
            markeredgecolor=c,
            markeredgewidth=1.5,
            markersize=7,
            linestyle=" ",
            label=str(pid),
        )
        for pid, c in pid_colours.items()
    ]
    legend = ax.legend(
        handles=handles,
        title="Participants",
        loc="lower center",
        ncol=3,
        frameon=True,
        facecolor="#FFEEB330",
        edgecolor="#FFEEB330",
        fontsize="small",
        bbox_to_anchor=(0.5, 0.10),
    )
    legend.get_frame().set_alpha(0.3)

    ax.set_ylim(0, ymax)
    tick_step = max(0.01, round(blups["amplitude_mean"].max() / 4, 2))
    rticks = np.arange(0, blups["amplitude_mean"].max() + tick_step * 2, tick_step)
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{v:.2f}" for v in rticks])
    ax.set_ylabel("Amplitude [mm]", labelpad=35)

    ax.set_xlim(0, 2 * np.pi)
    ax.xaxis.set_major_locator(
        plt.FixedLocator(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    )
    ax.xaxis.set_minor_locator(
        plt.FixedLocator(np.linspace(0, 2 * np.pi, 16, endpoint=False))
    )
    xtick_labels = [
        f"{t:.2f}\n({t / (2 * np.pi) * 24:.0f}:00)"
        for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ]
    ax.set_xticklabels(xtick_labels)
    ax.xaxis.set_tick_params(pad=7)

    ax.yaxis.grid(True, which="major", color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.xaxis.grid(True, which="major", color="black", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.xaxis.grid(True, which="minor", color="black", linestyle="-", linewidth=0.3, alpha=0.2)
    ax.set_rlabel_position(22.5)

    ax.set_title("Amplitude and phase by participant", pad=10, loc="left")
    offset = offset_copy(ax.transAxes, fig=fig, x=-2, units="points")
    ax.text(
        -0.15, 1.19, s=panel_label,
        transform=offset, fontsize=12, fontweight="bold", va="center", ha="right",
    )
    return ax


# ---------------------------------------------------------------------------
# Panel C: example trajectories with posterior fit
# ---------------------------------------------------------------------------

def _cosinor_curve(
    t_fine: np.ndarray,
    intercept: float,
    alpha_sin: float,
    beta_cos: float,
    tau: float,
) -> np.ndarray:
    """Cosinor curve with α on the sine term and β on the cosine term."""
    omega = 2 * np.pi / tau
    return intercept + alpha_sin * np.sin(omega * t_fine) + beta_cos * np.cos(omega * t_fine)


def plot_two_trajectories(
    fig: plt.Figure,
    ax: plt.Axes,
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    blups: pd.DataFrame,
    pid1: int,
    pid2: int,
    panel_label: str = "C",
) -> plt.Axes:
    ax.axhline(y=0, color="black", linestyle=(0, (5, 10)), linewidth=0.8, alpha=0.6)

    for pid, fc, ec, lc in [
        (pid1, "#6D94BC", "#3C68AB", "#30476B"),
        (pid2, "#D78DAD", "#C94E98", "#990033"),
    ]:
        d = data[data["participant_id"] == pid].sort_values("hours_since_start")
        p = predictions[predictions["participant_id"] == pid].sort_values("hours_since_start")
        b = blups[blups["participant_id"] == pid].iloc[0]
        tau = d["saliva_tau_hours"].iloc[0]

        ax.scatter(
            d["hours_since_start"], d["amplitude"],
            s=22, color=fc, edgecolor=ec, linewidth=0.8, zorder=3,
            label=f"Observed ({pid})",
        )

        t_fine = np.linspace(d["hours_since_start"].min(), d["hours_since_start"].max(), 300)
        y_fit = _cosinor_curve(t_fine, b["intercept_mean"], b["alpha_mean"], b["beta_mean"], tau)
        ax.plot(t_fine, y_fit, color=lc, linewidth=1.8, label=f"Cosinor fit ({pid})")

        ax.fill_between(
            p["hours_since_start"], p["mu_cond_lo"], p["mu_cond_hi"],
            color=lc, alpha=0.12, label="95% ETI",
        )

        # Convention y = μ + A·cos(ωt + φ): the peak sits at t = -φ/ω,
        # wrapped into one period so the marker is forward-looking.
        peak_hours = np.mod(-(b["phase_mean"] / (2 * np.pi)) * tau, tau)
        ax.axvline(x=peak_hours, color=fc, linestyle="--", linewidth=0.9, alpha=0.8)
        ax.text(
            x=data["hours_since_start"].max() + 0.5,
            y=d["amplitude"].max(),
            s=str(pid), color=lc, fontsize=11,
        )

    ax.set_xlabel("Experimental block")
    ax.set_ylabel("Amplitude [mm]", labelpad=10)

    n_blocks = int(data["hours_since_start"].max() / BLOCK_INTERVAL_HOURS) + 1
    ax.set_xticks(
        np.arange(0, n_blocks * BLOCK_INTERVAL_HOURS, BLOCK_INTERVAL_HOURS),
        np.arange(1, n_blocks + 1),
    )
    ax.xaxis.grid(True, which="major", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(True, which="major", color="gray", linestyle="-", linewidth=0.5, alpha=0.3)

    secax = ax.secondary_xaxis("top")
    max_h = int(data["hours_since_start"].max()) + 1
    secax.set_xticks(
        list(range(0, max_h, 8)),
        [f"{h:02d}:00" for h in range(0, max_h, 8)],
    )
    secax.set_xlabel("Hours since start [hh:mm]")

    ax.set_title(
        f"Example trajectories -- participants {pid1} and {pid2}",
        pad=12, loc="left",
    )
    offset = offset_copy(ax.transAxes, fig=fig, x=-2, units="points")
    ax.text(
        -0.048, 1.18, s=panel_label,
        transform=offset, fontsize=12, fontweight="bold", va="center", ha="right",
    )
    return ax


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def create_overview_figure(
    data: pd.DataFrame,
    blups: pd.DataFrame,
    predictions: pd.DataFrame,
    pid1: int,
    pid2: int,
    condition: str,
) -> tuple[plt.Figure, tuple]:
    fig = plt.figure(figsize=(8.5, 9), constrained_layout=True)
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1.1],
        width_ratios=[1, 1],
        hspace=0.05,
        wspace=0.05,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1 = plot_outcome_histogram(
        fig=fig, ax=ax1, data=data,
        y_col="amplitude",
        xlabel="Amplitude [mm]",
        panel_label="A",
    )

    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    ax2 = plot_polar_circadian(fig=fig, ax=ax2, blups=blups, panel_label="B")

    ax3 = fig.add_subplot(gs[1, :])
    ax3 = plot_two_trajectories(
        fig=fig, ax=ax3,
        data=data, predictions=predictions, blups=blups,
        pid1=pid1, pid2=pid2,
        panel_label="C",
    )

    cond_full = CONDITION_LABELS.get(condition, condition)
    fig.suptitle(
        f"Circadian modulation of pupil response amplitude (mm) -- {cond_full}",
        fontsize=11, y=1.01,
    )
    return fig, (ax1, ax2, ax3)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(condition: str = "flux", pid1: int = 212, pid2: int = 213) -> None:
    data, blups, predictions = load_condition_data(condition)

    out_dir = FIG_DIR / condition
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = create_overview_figure(
        data=data,
        blups=blups,
        predictions=predictions,
        pid1=pid1,
        pid2=pid2,
        condition=condition,
    )
    out_path = out_dir / f"fig_circadian_overview_mm_{condition}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Circadian pupil overview figure")
    parser.add_argument("--condition", default="flux", choices=list(CONDITION_LABELS.keys()))
    parser.add_argument("--pid1", type=int, default=204)
    parser.add_argument("--pid2", type=int, default=211)
    args = parser.parse_args()
    main(condition=args.condition, pid1=args.pid1, pid2=args.pid2)
