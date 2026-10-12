from pathlib import Path

import numpy as np
import pandas as pd

from civibeplr.vis_io import _in_jupyter, apply_font_style

apply_font_style()

_PROJECT_DIR = Path(__file__).resolve().parents[2]

from civibeplr.analysis_io import (
    PUPIL_PARTICIPANT_COL,
    PUPIL_CONDITION_COL,
    build_analysis_df,
    load_pupil_results,
    load_saliva_results,
    merge_pupil_saliva,
    get_parameter_label,
)


# ═══════════════════════════════════════════════════════════════════════════
# Outcome distribution inspection
# ═══════════════════════════════════════════════════════════════════════════

def inspect_parameter_distribution(
    analysis_df: pd.DataFrame,
    parameter_col: str = "amplitude",
    figure_dir: Path | None = None,
    kde: bool = False,
) -> pd.DataFrame:
    """Inspect per-condition normality of the outcome before model fitting.

    Prints Shapiro-Wilk W, p-value, skewness, and excess kurtosis for every
    condition.  Saves a figure with one row per condition: histogram (left)
    and Q-Q plot against Normal quantiles (right).

    Returns a DataFrame with per-condition summary statistics.
    """
    from scipy import stats
    import matplotlib.pyplot as plt

    if figure_dir is None:
        figure_dir = _PROJECT_DIR / "results" / "figures" / "circadian_cosinor"
    figure_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted(analysis_df[PUPIL_CONDITION_COL].dropna().unique())
    n_conds = len(conditions)

    fig, axes = plt.subplots(n_conds, 2, figsize=(10, 3 * n_conds), squeeze=False)

    summary_rows = []
    print(f"\Parameter distribution check: '{parameter_col}'")
    header = f"{'Condition':<12} {'N':>5} {'Mean':>8} {'SD':>8} {'Skew':>8} {'Kurt':>8} {'W':>8} {'p':>8}  Normal?"
    print(header)
    print("-" * len(header))

    for i, cond in enumerate(conditions):
        vals = (
            analysis_df.loc[analysis_df[PUPIL_CONDITION_COL] == cond, parameter_col]
            .dropna()
            .values
        )
        n = len(vals)
        mean = float(np.mean(vals))
        sd   = float(np.std(vals, ddof=1))
        skew = float(stats.skew(vals))
        kurt = float(stats.kurtosis(vals))  # excess kurtosis
        w_stat, p_val = stats.shapiro(vals)
        normal_str = "yes" if p_val >= 0.05 else "NO"

        print(f"{cond:<12} {n:>5} {mean:>8.3f} {sd:>8.3f} {skew:>8.3f} {kurt:>8.3f} "
              f"{w_stat:>8.4f} {p_val:>8.4f}  {normal_str}")

        summary_rows.append({
            "condition": cond, "n": n, "mean": mean, "sd": sd,
            "skewness": skew, "excess_kurtosis": kurt,
            "shapiro_W": w_stat, "shapiro_p": p_val,
        })

        ax_hist = axes[i, 0]
        ax_qq   = axes[i, 1]

        ax_hist.hist(vals, bins="auto", color="steelblue", edgecolor="white", alpha=0.85,
                     density=kde)
        ax_hist.set_title(f"{cond.upper()}")
        ax_hist.set_xlabel(get_parameter_label(parameter_col))
        ax_hist.set_ylabel("Density" if kde else "Count")

        if kde:
            kde_xs = np.linspace(vals.min(), vals.max(), 300)
            kde_ys = stats.gaussian_kde(vals)(kde_xs)
            ax_hist.plot(kde_xs, kde_ys, color="steelblue", lw=1.5, linestyle="--", label="KDE")
            normal_ys = stats.norm.pdf(kde_xs, loc=mean, scale=sd)
            ax_hist.plot(kde_xs, normal_ys, color="firebrick", lw=1.5, label="Normal")
            ax_hist.legend(fontsize=8)
        ax_hist.text(
            0.97, 0.95,
            f"skew={skew:.2f}  kurt={kurt:.2f}\nShapiro W={w_stat:.3f}, "
            + ("$p$<0.001" if p_val < 0.001 else f"$p$={p_val:.3f}"),
            transform=ax_hist.transAxes, ha="right", va="top", #fontsize=8,
        )

        (osm, osr), (slope, intercept, _) = stats.probplot(vals, dist="norm")
        ax_qq.scatter(osm, osr, s=20, color="steelblue", alpha=0.85)
        x_line = np.array([min(osm), max(osm)])
        ax_qq.plot(x_line, slope * x_line + intercept, color="firebrick", lw=1.5)
        ax_qq.set_title(f"{cond.upper()}: Q-Q plot")
        ax_qq.set_xlabel("Theoretical quantiles")
        ax_qq.set_ylabel("Sample quantiles")

    fig.suptitle(f"Parameter distribution: {get_parameter_label(parameter_col)}", fontsize=13)
    fig.tight_layout()
    fig_path = figure_dir / f"fig_parameter_distribution_{parameter_col}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    if not _in_jupyter():
        plt.close(fig)
    print(f"\n  Saved: {fig_path}")

    return pd.DataFrame(summary_rows)


def main(
    pupil_path: Path | None = None,
    saliva_path: Path | None = None,
    outcome_col: str = "amplitude",
) -> pd.DataFrame:
    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading data...")
    pupil_df = load_pupil_results(pupil_path)
    saliva_df = load_saliva_results(saliva_path)
    print(
        f"  Pupil:  {len(pupil_df)} rows  |  "
        f"{pupil_df[PUPIL_PARTICIPANT_COL].nunique()} participants  |  "
        f"{pupil_df[PUPIL_CONDITION_COL].nunique()} conditions"
    )
    print(
        f"  Saliva: {len(saliva_df)} participants  |  "
        f"τ range {saliva_df['tau_hours'].min():.2f}–{saliva_df['tau_hours'].max():.2f} h"
    )

    # ── Merge and prepare ─────────────────────────────────────────────────
    merged_df = merge_pupil_saliva(pupil_df, saliva_df)
    analysis_df = build_analysis_df(merged_df)

    # ── Inspect outcome distribution before modelling ──────────────────────
    inspect_parameter_distribution(analysis_df, parameter_col=outcome_col)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect per-condition amplitude distributions before model fitting."
    )
    parser.add_argument(
        "--outcome",
        default=None,
        choices=["amplitude", "log_amplitude", "amplitude_2f", "log_amplitude_2f"],
        help="Single outcome column to inspect. Omit to inspect all four.",
    )
    parser.add_argument(
        "--kde", action="store_true",
        help="Overlay a KDE curve on the histograms (switches y-axis to density).",
    )
    args = parser.parse_args()

    print("Loading data...")
    pupil_df  = load_pupil_results()
    saliva_df = load_saliva_results()
    merged_df = merge_pupil_saliva(pupil_df, saliva_df)
    analysis_df = build_analysis_df(merged_df)

    cols = (
        [args.outcome]
        if args.outcome
        else ["amplitude", "log_amplitude", "amplitude_2f", "log_amplitude_2f", "baseline", "phase", "phase_2f", "tonic_min"]
    )
    for col in cols:
        inspect_parameter_distribution(analysis_df, parameter_col=col, kde=True)#args.kde)
