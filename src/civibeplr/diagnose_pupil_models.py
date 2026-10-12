"""Diagnostic plots for the two-step cosinor pupil model.

Loads the fitted parameters CSV produced by ``fit_pupil_trace_2step_cosinor_model``
and generates two diagnostic figures:

1. Tonic scatter — Step 1 tonic minimum vs. tonic mean across conditions.
2. R² distribution — distribution of Step 2 cosinor goodness-of-fit.

Requires ``fit_pupil_trace_2step_cosinor_model`` to have been run first.

Usage
-----
python -m civibeplr.diagnose_pupil_models
python -m civibeplr.diagnose_pupil_models --results path/to/pupil_cosinor_2step_param.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from scipy.stats import pearsonr

from civibeplr.data_io import DEFAULT_CONDITION_COL
from civibeplr.latex_table_creation import format_vars_pupil_cosinor_to_latex
from civibeplr.vis_io import TRIAL_TYPE_COLORS, _in_jupyter, apply_font_style

apply_font_style()

# ── Default paths ─────────────────────────────────────────────────────────────

_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = _PROJECT_DIR / "results" 
DEFAULT_RESULTS_CSV = (
    _PROJECT_DIR / "results" / "tables" / "pupil_cosinor" / "pupil_cosinor_2step_param.csv"
)
DEFAULT_FIG_DIR = _PROJECT_DIR / "results" / "figures" / "pupil_cosinor_diagnostics"


def _draw_tonic_scatter_panel(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    x_span: float,
    y_span: float,
    marginal_fraction: float = 0.25,
) -> bool:
    """Draw tonic_mean-vs-tonic_min scatter with marginal histograms onto `ax`.

    Returns False (and skips the marginals/ticks/limits) when there are fewer than
    3 points, since a correlation and marginal histograms are not meaningful below that.
    """
    ax.scatter(x, y, color=color, alpha=0.6, s=20, edgecolors="none", zorder=3)
    ax.grid(True, ls="-", lw=0.3, alpha=0.5, zorder=0)

    if len(x) >= 3:
        r, _p = pearsonr(x, y)
        ax.text(
            0.97, 0.97, f"r = {r:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, family="monospace",
        )

    if len(x) < 3:
        return False

    # Bottom marginal: histogram of tonic_min (amplitude), pointing upward from y0
    counts_x, edges_x = np.histogram(x, bins=15, range=(x0, x1))
    bin_w_x = edges_x[1] - edges_x[0]
    bar_h = counts_x / counts_x.max() * marginal_fraction * y_span
    ax.bar(edges_x[:-1], bar_h, width=bin_w_x * 0.85, bottom=y0,
           align="edge", color=color, alpha=0.45, edgecolor="none", zorder=2)

    # Left marginal: histogram of tonic_mean, pointing rightward from x0
    counts_y, edges_y = np.histogram(y, bins=15, range=(y0, y1))
    bin_w_y = edges_y[1] - edges_y[0]
    bar_w = counts_y / counts_y.max() * marginal_fraction * x_span
    ax.barh(edges_y[:-1], bar_w, height=bin_w_y * 0.85, left=x0,
            align="edge", color=color, alpha=0.45, edgecolor="none", zorder=2)

    # Increase number of steps between current ticks (not limits) for better grid alignment with the scatter points
    ax.set_xticks(np.linspace(ax.get_xticks()[0], ax.get_xticks()[-1], num=6))
    ax.set_yticks(np.linspace(ax.get_yticks()[0], ax.get_yticks()[-1], num=6))
    # Set limits after drawing so bar artists don't expand the view
    ax.set_xlim(x0, 0.8)
    ax.set_ylim(y0, y1)
    return True


def _draw_rsquared_panel(ax, r2_vals: pd.Series, color: str) -> None:
    """Draw an R² histogram with a median line onto `ax`."""
    ax.hist(r2_vals, bins=20, color=color, edgecolor="none", rwidth=0.85, alpha=0.6)

    median_r2 = r2_vals.median()
    ax.axvline(
        median_r2, color=color, lw=1.2, ls="--", label=f"Median = {median_r2:.3f}"
    )
    ax.legend(fontsize=8, frameon=False)

    ax.grid(True, axis="y", ls="-", lw=0.3, alpha=0.5)


def plot_step1_tonic_scatter(
    results_df: pd.DataFrame,
    condition_col: str = DEFAULT_CONDITION_COL,
    output_path: Path | None = None,
    marginal_fraction: float = 0.25,
) -> None:
    """Scatter of tonic_mean (y) vs tonic_min (x) with in-plot marginal histograms.

    Layout: 2×3 grid (5 condition panels + 1 empty cell). Histogram distributions are
    drawn inside each scatter panel: bottom edge for tonic_min (amplitude), left edge
    for tonic_mean, each scaled to `marginal_fraction` of the respective axis range.
    All panels share the same axis limits based on the global data range.
    """
    conditions = sorted(results_df[condition_col].dropna().unique())
    n_cols = 3
    n_rows = 2

    # Shared axis limits across all panels
    valid = results_df[["tonic_mean", "tonic_min"]].dropna()
    x_pad = 0.05 * (valid["tonic_min"].max() - valid["tonic_min"].min() or 1.0)
    y_pad = 0.05 * (valid["tonic_mean"].max() - valid["tonic_mean"].min() or 1.0)
    x0 = valid["tonic_min"].min() - x_pad
    x1 = valid["tonic_min"].max() + x_pad
    y0 = valid["tonic_mean"].min() - y_pad
    y1 = valid["tonic_mean"].max() + y_pad
    x_span, y_span = x1 - x0, y1 - y0

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols, 2.0 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i_cond, cond in enumerate(conditions):
        ax = axes[i_cond // n_cols, i_cond % n_cols]
        color = TRIAL_TYPE_COLORS.get(cond, "tab:blue")
        cond_data = results_df.loc[
            results_df[condition_col] == cond, ["tonic_mean", "tonic_min"]
        ].dropna()
        x = cond_data["tonic_min"].to_numpy()
        y = cond_data["tonic_mean"].to_numpy()

        if not _draw_tonic_scatter_panel(
            ax, x, y, color, x0, x1, y0, y1, x_span, y_span, marginal_fraction
        ):
            continue

        ax.set_title(cond.upper())
        # Leftmost panels get y-label, top panels get x-labels
        if i_cond % n_cols == 0 :
            ax.set_ylabel("Tonic mean [mm]")
        if i_cond // n_cols == n_rows - 1:
            ax.set_xlabel("Tonic amplitude [mm]")

    axes[1, 2].set_visible(False)

    fig.suptitle("Step 1: Tonic pupil response", y=1.002)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure in: {output_path}")
        if not _in_jupyter():
            plt.close(fig)
    else:
        plt.show()


def plot_step2_rsquared_distribution(
    results_df: pd.DataFrame,
    condition_col: str = DEFAULT_CONDITION_COL,
    output_path: Path | None = None,
) -> None:
    """Histogram of R² from the cosinor-on-residual fit, one panel per condition.

    Layout: 2×3 grid (5 condition panels + 1 empty cell).
    """
    conditions = sorted(results_df[condition_col].dropna().unique())
    n_cols = 3
    n_rows = 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.5 * n_cols, 2.0 * n_rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i_cond, cond in enumerate(conditions):
        ax = axes[i_cond // n_cols, i_cond % n_cols]
        color = TRIAL_TYPE_COLORS.get(cond, "tab:blue")
        r2_vals = results_df.loc[results_df[condition_col] == cond, "r_squared"].dropna()

        _draw_rsquared_panel(ax, r2_vals, color)

        ax.set_title(cond.upper())
        # Leftmost panels get y-label, top panels get x-labels
        if i_cond % n_cols == 0:
            ax.set_ylabel("Count")
        if i_cond // n_cols == n_rows - 1:
            ax.set_xlabel("R²")
        #
        ax.tick_params()

    axes[1, 2].set_visible(False)

    fig.suptitle("Step 2: Goodness-of-fit (R²) distribution", y=1.002)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure in: {output_path}")
        if not _in_jupyter():
            plt.close(fig)
    else:
        plt.show()


def plot_pupil_cosinor_diagnostics_grid(
    results_df: pd.DataFrame,
    condition_col: str = DEFAULT_CONDITION_COL,
    output_path: Path | None = None,
    marginal_fraction: float = 0.25,
) -> None:
    """Combined Step 1/Step 2 diagnostics: one row per condition, tonic scatter next to R² histogram.

    Column 0 reproduces `plot_step1_tonic_scatter`'s panels (tonic mean vs. tonic amplitude
    with marginal histograms); column 1 reproduces `plot_step2_rsquared_distribution`'s panels
    (R² histograms with a median line).
    """
    conditions = sorted(results_df[condition_col].dropna().unique())
    n_rows = len(conditions)

    # Shared axis limits for the scatter column, as in plot_step1_tonic_scatter
    valid = results_df[["tonic_mean", "tonic_min"]].dropna()
    x_pad = 0.05 * (valid["tonic_min"].max() - valid["tonic_min"].min() or 1.0)
    y_pad = 0.05 * (valid["tonic_mean"].max() - valid["tonic_mean"].min() or 1.0)
    x0 = valid["tonic_min"].min() - x_pad
    x1 = valid["tonic_min"].max() + x_pad
    y0 = valid["tonic_mean"].min() - y_pad
    y1 = valid["tonic_mean"].max() + y_pad
    x_span, y_span = x1 - x0, y1 - y0

    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(5.0, 1.4 * n_rows),
        sharex="col",
        sharey="col",
        squeeze=False,
        constrained_layout=True,
    )

    for i_cond, cond in enumerate(conditions):
        color = TRIAL_TYPE_COLORS.get(cond, "tab:blue")
        is_last_row = i_cond == n_rows - 1

        # Column 0: tonic scatter
        ax_scatter = axes[i_cond, 0]
        cond_tonic = results_df.loc[
            results_df[condition_col] == cond, ["tonic_mean", "tonic_min"]
        ].dropna()
        x = cond_tonic["tonic_min"].to_numpy()
        y = cond_tonic["tonic_mean"].to_numpy()
        _draw_tonic_scatter_panel(
            ax_scatter, x, y, color, x0, x1, y0, y1, x_span, y_span, marginal_fraction
        )
        ax_scatter.set_ylabel("Tonic mean [mm]")
        ax_scatter.annotate(
            cond.upper(),
            xy=(0, 0.5), xytext=(-ax_scatter.yaxis.labelpad - 16, 0),
            xycoords=ax_scatter.yaxis.label, textcoords="offset points",
            ha="right", va="center", fontsize=10, #fontweight="bold", rotation=90,
        )
        if is_last_row:
            ax_scatter.set_xlabel("Tonic amplitude [mm]", labelpad=2)

        # Column 1: R² distribution
        ax_hist = axes[i_cond, 1]
        r2_vals = results_df.loc[results_df[condition_col] == cond, "r_squared"].dropna()
        _draw_rsquared_panel(ax_hist, r2_vals, color)
        ax_hist.set_ylabel("Count")
        if is_last_row:
            ax_hist.set_xlabel("R²", labelpad=2)

    axes[0, 0].set_title("Tonic pupil response", pad=12)
    axes[0, 1].set_title("Goodness-of-fit (R²)", pad=12)

    # Condition color key, mirroring the legend style used in fig_global_condition_means
    legend_handles = [
        Line2D([], [], color=TRIAL_TYPE_COLORS.get(cond, "tab:blue"),
               marker="o", linestyle="none", markersize=5, label=cond.upper())
        for cond in conditions
    ]
    fig.legend(
        handles=legend_handles, loc="outside lower center",
        ncol=len(legend_handles), fontsize=8, frameon=False,
    )

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved Figure in: {output_path}")
        if not _in_jupyter():
            plt.close(fig)
    else:
        plt.show()


def calculate_tonic_correlation(results_df: pd.DataFrame) -> dict:
    """
    Calculate the Pearson correlation between tonic_min (amplitude) and tonic_mean for each condition.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing 'tonic_min', 'tonic_mean', and condition columns.
    condition_col : str
        Name of the condition column (default: DEFAULT_CONDITION_COL).

    Returns
    -------
    dict
        Dictionary mapping condition to a dict with keys 'r', 'p', 'n', 'valid'.
    """
    from scipy.stats import pearsonr
    out = {}
    condition_col = DEFAULT_CONDITION_COL
    for cond in sorted(results_df[condition_col].dropna().unique()):
        sub = results_df[results_df[condition_col] == cond][["tonic_min", "tonic_mean"]].dropna()
        n = len(sub)
        if n < 3:
            out[f"r_tonic_mean_min_{cond}"] = {"r": None, "p": None, "n": n, "valid": False}
        else:
            r, p = pearsonr(sub["tonic_min"], sub["tonic_mean"])
            out[f"r_tonic_mean_min_{cond}"] = {"r": r, "p": p, "n": n, "valid": True}
    return out


def format_vars_tonic_correlation_to_latex(correlation_dict: dict, save_tex: bool) -> None:
    """Write LaTeX \\def variables for tonic correlation results to a .tex file.
    
    Parameters
    ----------
    correlation_dict : dict
        Dictionary mapping condition to a dict with keys 'r', 'p', 'n', 'valid'.
    save_tex : bool
        Whether to save the .tex file. If False, the function will print the intended file path and return it without writing.  
    """
    from civibeplr.latex_table_creation import _to_camel

    tex_dir = DEFAULT_RESULT_DIR / "tex"
    tex_dir.mkdir(parents=False, exist_ok=True)
    filepath = tex_dir / "pupil_cosinor_diagnostics" / "vars_tonic_correlation_results.tex"
    filepath.parent.mkdir(parents=False, exist_ok=True)

    if not save_tex:
        print("LaTeX output not saved (save_tex=False). Returning file path for reference:")
    if save_tex:
        lines = ["% LaTeX \\def variables for tonic_mean vs tonic_min correlations by condition"]
        for key, stats in correlation_dict.items():
            var = _to_camel(key)
            r_val = stats["r"]
            if r_val is not None:
                lines.append(rf"\def\{var}{{{r_val:.2f}}}")
        if filepath.exists():
            print(f"Overwriting existing file: {filepath}")
        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nSaved correlations to: {filepath}")

    return filepath


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def run_diagnostics(results_csv: Path, fig_dir: Path) -> None:
    if not results_csv.exists():
        raise FileNotFoundError(
            f"Results CSV not found: {results_csv}\n"
            "Run fit_pupil_trace_2step_cosinor_model first."
        )

    results_df = pd.read_csv(results_csv)
    fig_dir.mkdir(parents=False, exist_ok=True)

    plot_pupil_cosinor_diagnostics_grid(
        results_df,
        output_path=fig_dir / "fig_pupil_cosinor_diagnostics_combined.png",
    )
    correlation_dict = calculate_tonic_correlation(results_df)
    format_vars_tonic_correlation_to_latex(correlation_dict, save_tex=True)
    format_vars_pupil_cosinor_to_latex(results_df)

    return correlation_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnostic plots for the two-step cosinor pupil model."
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"Path to fitted parameters CSV (default: {DEFAULT_RESULTS_CSV}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIG_DIR,
        help=f"Directory for output figures (default: {DEFAULT_FIG_DIR}).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_diagnostics(results_csv=args.results, fig_dir=args.output_dir)
