"""Prespecified sensitivity analyses for the registered pupil outcome.

Two analyses, both listed in the locked decisions before any model was fitted
(``todo/protocol_plr_supplementary_analysis_plan.md``):

1. **Leave-one-participant-out, L-M only.**  L-M is the condition whose
   evidence sits near the decision boundary (log10 BF10 = -0.32), so it is the
   one where a single participant could change which model the data favour;
   holding one out moves it between -0.88 and +0.38.  The other conditions
   were not checked, and no claim about their participant influence is made.

2. **Registered period range.**  The protocol registered a tau scan over
   20-28 h; the pipeline uses 22-28 h.

   Caveat, and it is a real one: the protocol also registered a *single*
   harmonic, whereas ``fit_saliva_cosinor_model`` fits the fundamental and the
   second harmonic together.  Reconstructing that would mean changing the
   primary saliva module, which this branch does not do.  So this analysis
   isolates the period *range* only, and the supplement must say so rather
   than claim the registered period procedure was restored.

The log-ratio sensitivity is not here: it is a plain outcome choice, run via
``--outcome log_plr_ratio`` through the ordinary pipeline.

Outputs land under ``results/tables/plr_ratio/sensitivity/``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from civibeplr.analyse_circadian_pupil_parameters import (
    DEFAULT_SMC_DRAWS,
    build_models,
    compute_log_bf10,
    fit_condition,
)
from civibeplr.si_analyse_plr_ratio import (
    DEFAULT_SEED,
    add_circadian_regressors,
    prepare_analysis_df,
)
from civibeplr.analysis_io import (
    ANALYSIS_PARTICIPANT_COL,
    PUPIL_CONDITION_COL,
    merge_pupil_saliva,
)
from civibeplr.si_derive_protocol_plr import (
    DEFAULT_DERIV_DIR,
    NON_FLUX_CONDITIONS,
)

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SENS_DIR = (
    DEFAULT_PROJECT_DIR / "results" / "tables" / "plr_ratio" / "sensitivity"
)
DEFAULT_REGISTERED_PERIOD_DIR = (
    DEFAULT_PROJECT_DIR / "data" / "derivatives" / "plr_ratio" / "registered_period"
)

REGISTERED_PERIOD_MIN = 20.0
REGISTERED_PERIOD_MAX = 28.0


def _fit_log_bf10(
    df_cond: pd.DataFrame,
    outcome_col: str,
    smc_draws: int,
    seed: int,
) -> float:
    """Fit M0/M1 on one condition subset and return log10 BF10."""
    m0, m1 = build_models(df_cond, outcome_col=outcome_col)
    idata_m0, idata_m1 = fit_condition(m0, m1, smc_draws=smc_draws, random_seed=seed)
    return compute_log_bf10(idata_m0, idata_m1)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Leave-one-participant-out
# ═══════════════════════════════════════════════════════════════════════════

#: Only L-M is reported: see the module docstring.
LOPO_CONDITIONS = ["l-m"]


def leave_one_participant_out(
    df: pd.DataFrame,
    outcome_col: str = "plr_ratio",
    conditions: list[str] | None = None,
    smc_draws: int = DEFAULT_SMC_DRAWS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Refit every condition with each participant held out in turn.

    Returns one row per (condition, held-out participant) plus a ``full`` row
    per condition, with the Bayes factor and the shift relative to full.
    """
    conditions = conditions or LOPO_CONDITIONS
    rows: list[dict[str, object]] = []

    for cond in conditions:
        df_cond = df[df[PUPIL_CONDITION_COL] == cond].dropna(subset=[outcome_col])
        participants = sorted(df_cond[ANALYSIS_PARTICIPANT_COL].unique())

        print(f"\n[{cond}] full fit (n={len(df_cond)}, {len(participants)} participants)")
        full_bf = _fit_log_bf10(df_cond, outcome_col, smc_draws, seed)
        rows.append(
            {
                "condition": cond,
                "held_out": "none",
                "n_obs": len(df_cond),
                "n_participants": len(participants),
                "log_bf10": full_bf,
                "delta_vs_full": 0.0,
            }
        )
        print(f"  full log10 BF10 = {full_bf:+.3f}")

        for pid in participants:
            sub = df_cond[df_cond[ANALYSIS_PARTICIPANT_COL] != pid]
            print(f"[{cond}] holding out participant {pid} (n={len(sub)})")
            bf = _fit_log_bf10(sub, outcome_col, smc_draws, seed)
            rows.append(
                {
                    "condition": cond,
                    "held_out": str(pid),
                    "n_obs": len(sub),
                    "n_participants": sub[ANALYSIS_PARTICIPANT_COL].nunique(),
                    "log_bf10": bf,
                    "delta_vs_full": bf - full_bf,
                }
            )
            print(f"  log10 BF10 = {bf:+.3f}  (delta {bf - full_bf:+.3f})")

    return pd.DataFrame(rows)


def summarise_lopo(lopo: pd.DataFrame) -> pd.DataFrame:
    """Per-condition summary: does any single participant flip the conclusion?

    ``crosses_zero`` flags a condition where holding out one participant moves
    log10 BF10 across zero, i.e. changes which model the data favour.
    """
    rows = []
    for cond, g in lopo.groupby("condition"):
        full = g[g["held_out"] == "none"]["log_bf10"].iloc[0]
        folds = g[g["held_out"] != "none"]
        worst = folds.loc[folds["delta_vs_full"].abs().idxmax()]
        rows.append(
            {
                "condition": cond,
                "log_bf10_full": full,
                "log_bf10_min": folds["log_bf10"].min(),
                "log_bf10_max": folds["log_bf10"].max(),
                "max_abs_delta": folds["delta_vs_full"].abs().max(),
                "most_influential": worst["held_out"],
                "crosses_zero": bool(
                    (np.sign(folds["log_bf10"]) != np.sign(full)).any()
                ),
            }
        )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Registered period range
# ═══════════════════════════════════════════════════════════════════════════

def build_registered_period_ratios(
    saliva_output_dir: Path | None = None,
) -> pd.DataFrame:
    """Rebuild circadian regressors from a 20-28 h tau scan.

    Re-runs ``fit_saliva_cosinor_model.main()`` into a *separate* output
    directory so the primary saliva results are untouched.

    This restores the registered period *range* only.  The registered model was
    single-harmonic; the pipeline's is two-harmonic, and changing that would
    require editing the primary module.
    """
    from civibeplr.fit_saliva_cosinor_model import main as fit_saliva

    saliva_output_dir = saliva_output_dir or DEFAULT_REGISTERED_PERIOD_DIR
    # fit_saliva writes tables, tex and figures into fixed subdirectories but
    # only creates some of them, so pre-create all three.
    for sub in ("tables", "tex", "figures"):
        (saliva_output_dir / sub / "saliva_cosinor").mkdir(parents=True, exist_ok=True)

    print(f"Re-fitting saliva cosinor over tau in "
          f"[{REGISTERED_PERIOD_MIN}, {REGISTERED_PERIOD_MAX}] h ...")
    fit_saliva(
        period_min=REGISTERED_PERIOD_MIN,
        period_max=REGISTERED_PERIOD_MAX,
        output_dir=saliva_output_dir,
    )

    best_fit = saliva_output_dir / "tables" / "saliva_cosinor" / "saliva_cosinor_best_fit.csv"
    if not best_fit.exists():
        candidates = list(saliva_output_dir.rglob("saliva_cosinor_best_fit.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"Registered-period saliva fit not found under {saliva_output_dir}"
            )
        best_fit = candidates[0]

    ratios = pd.read_csv(DEFAULT_DERIV_DIR / "paired_ratios_analysis.csv")
    merged = merge_pupil_saliva(ratios, pd.read_csv(best_fit))
    merged = merged.dropna(subset=["saliva_tau_hours"])
    return add_circadian_regressors(merged)


def compare_taus(registered_df: pd.DataFrame, primary_df: pd.DataFrame) -> pd.DataFrame:
    """Per-participant tau under the registered range vs the pipeline range."""
    reg = (
        registered_df.groupby(ANALYSIS_PARTICIPANT_COL)["saliva_tau_hours"]
        .first()
        .rename("tau_registered_20_28")
    )
    pri = (
        primary_df.groupby(ANALYSIS_PARTICIPANT_COL)["saliva_tau_hours"]
        .first()
        .rename("tau_primary_22_28")
    )
    out = pd.concat([pri, reg], axis=1).reset_index()
    out["delta_hours"] = out["tau_registered_20_28"] - out["tau_primary_22_28"]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def fit_all_conditions(
    df: pd.DataFrame,
    label: str,
    outcome_col: str = "plr_ratio",
    smc_draws: int = DEFAULT_SMC_DRAWS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Fit every non-FLUX condition on a prepared frame and return BFs."""
    rows = []
    for cond in sorted(df[PUPIL_CONDITION_COL].dropna().unique()):
        if cond not in NON_FLUX_CONDITIONS:
            raise AssertionError(f"Unexpected condition: {cond}")
        df_cond = df[df[PUPIL_CONDITION_COL] == cond].dropna(subset=[outcome_col])
        print(f"\n[{label} | {cond}] n={len(df_cond)}  "
              f"participants={df_cond[ANALYSIS_PARTICIPANT_COL].nunique()}")
        bf = _fit_log_bf10(df_cond, outcome_col, smc_draws, seed)
        print(f"  log10 BF10 = {bf:+.3f}")
        rows.append(
            {
                "analysis": label,
                "condition": cond,
                "n_obs": len(df_cond),
                "n_participants": int(df_cond[ANALYSIS_PARTICIPANT_COL].nunique()),
                "log_bf10": bf,
            }
        )
    return pd.DataFrame(rows)


def main(
    analyses: list[str] | None = None,
    outcome_col: str = "plr_ratio",
    smc_draws: int = DEFAULT_SMC_DRAWS,
    seed: int = DEFAULT_SEED,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the prespecified sensitivity analyses."""
    warnings.filterwarnings("ignore")
    analyses = analyses or ["lopo", "registered-period"]
    output_dir = output_dir or DEFAULT_SENS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    primary = prepare_analysis_df()
    out: dict[str, pd.DataFrame] = {}

    if "lopo" in analyses:
        print("\n" + "=" * 70)
        print(f"PARTICIPANT INFLUENCE -- LOPO ({', '.join(LOPO_CONDITIONS)})")
        print("=" * 70)
        lopo = leave_one_participant_out(
            primary, outcome_col=outcome_col, smc_draws=smc_draws, seed=seed
        )
        summary = summarise_lopo(lopo)
        tag = "_".join(c.replace("-", "") for c in LOPO_CONDITIONS)
        lopo.to_csv(output_dir / f"lopo_{tag}_{outcome_col}.csv", index=False)
        summary.to_csv(output_dir / f"lopo_{tag}_summary_{outcome_col}.csv", index=False)
        out["lopo"] = lopo
        out["lopo_summary"] = summary
        print("\n" + summary.round(3).to_string(index=False))

    if "registered-period" in analyses:
        print("\n" + "=" * 70)
        print("PERIOD-RANGE SENSITIVITY -- registered 20-28 h range")
        print("=" * 70)
        df_reg = build_registered_period_ratios()
        taus = compare_taus(df_reg, primary)
        taus.to_csv(output_dir / "registered_period_taus.csv", index=False)
        print("\nPer-participant tau:")
        print(taus.round(2).to_string(index=False))

        res = fit_all_conditions(
            df_reg, "registered_period_20_28", outcome_col, smc_draws, seed
        )
        res.to_csv(output_dir / "registered_period.csv", index=False)
        out["registered_period"] = res
        out["taus"] = taus
        print("\n" + res.round(3).to_string(index=False))

    print(f"\nWrote sensitivity outputs to {output_dir}")
    return out


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Sensitivity analyses for the protocol PLR outcome.")
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=["lopo", "registered-period"],
        default=None,
    )
    parser.add_argument("--outcome", default="plr_ratio")
    parser.add_argument("--smc-draws", type=int, default=DEFAULT_SMC_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    main(
        analyses=args.analyses,
        outcome_col=args.outcome,
        smc_draws=args.smc_draws,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    _cli()
