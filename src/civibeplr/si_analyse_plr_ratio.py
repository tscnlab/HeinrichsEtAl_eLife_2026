"""Circadian models for the registered FLUX-normalised pupil outcome.

Fits the manuscript's M0/M1 pair to ``plr_ratio`` derived by
``si_derive_protocol_plr``, for the four non-FLUX conditions.  FLUX is not
modelled: FLUX/FLUX is 1 by construction.

The modelling primitives are imported unchanged from
``analyse_circadian_pupil_parameters``.  ``build_models()`` is already
outcome-agnostic -- its priors scale to the outcome's empirical SD -- so it
transfers to a dimensionless ratio without modification.  Its ``main()`` is not
reusable: it is bound to ``load_pupil_results()``, ``build_analysis_df()`` and
hard-coded ``amplitude``/``log_amplitude`` columns.

``build_analysis_df()`` is deliberately *not* refactored.  It hard-codes
``amplitude``, ``baseline_mean``, ``phase_rad`` and ``phase_2f_rad`` and would
raise on a ratio frame, so this module defines its own small regressor helper.
No existing module is modified by this branch, which makes "primary pipeline
outputs unchanged" provable by inspection rather than by regression test.

Outputs use the primary pipeline's layout, keyed by outcome name exactly as
``amplitude`` or ``tonic_min`` are (``circadian_cosinor/plr_ratio/`` and so on).
The outcome names cannot collide with any primary outcome, so the primary
results are never overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from civibeplr.analysis_io import (
    ANALYSIS_PARTICIPANT_COL,
    BLOCK_INTERVAL_HOURS,
    PUPIL_BLOCK_COL,
    PUPIL_CONDITION_COL,
    load_saliva_results,
    merge_pupil_saliva,
)
from civibeplr.analyse_circadian_pupil_parameters import (
    DEFAULT_IDATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SMC_DRAWS,
    build_models,
    extract_results,
    fit_condition,
    load_idata,
    save_idata,
    save_results,
)
from civibeplr.diagnose_circadian_models import run_diagnostics
from civibeplr.inspect_circadian_model import inspect_parameter_distribution
from civibeplr.si_derive_protocol_plr import (
    DEFAULT_DERIV_DIR,
    NON_FLUX_CONDITIONS,
)

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_ANALYSIS_CSV = DEFAULT_DERIV_DIR / "paired_ratios_analysis.csv"

#: Seed used for the manuscript analysis; see CLAUDE.md.
DEFAULT_SEED = 1072

OUTCOME_CHOICES = ["plr_ratio", "log_plr_ratio"]


def add_circadian_regressors(df: pd.DataFrame) -> pd.DataFrame:
    """Add elapsed time and the sine/cosine circadian regressors.

    Reuses the manuscript definitions exactly:

        hours_since_start = 3.75 * block
        alpha_sin         = sin(2*pi*t / tau_i)
        beta_cos          = cos(2*pi*t / tau_i)

    where tau_i is the participant-specific period from the saliva cosinor.
    ``alpha_sin`` and ``beta_cos`` are the regressor columns; their slopes are
    alpha and beta.

    This is the ratio-outcome counterpart of ``build_analysis_df()``, kept
    local so that no existing module has to change.
    """
    df = df.copy()
    df["hours_since_start"] = df[PUPIL_BLOCK_COL] * BLOCK_INTERVAL_HOURS

    phase = 2.0 * np.pi * df["hours_since_start"] / df["saliva_tau_hours"]
    df["alpha_sin"] = np.sin(phase)
    df["beta_cos"] = np.cos(phase)

    # Log ratio, available for every observation: all ratios are strictly
    # positive in this dataset.  Reported as a prespecified sensitivity
    # analysis, never as a silent replacement for the registered raw ratio.
    if (df["plr_ratio"] > 0).all():
        df["log_plr_ratio"] = np.log(df["plr_ratio"])
    else:
        n_bad = int((df["plr_ratio"] <= 0).sum())
        print(
            f"  WARNING: {n_bad} non-positive ratio(s); log_plr_ratio set to NaN "
            "for those rows. The log sensitivity analysis is not available for them."
        )
        df["log_plr_ratio"] = np.log(df["plr_ratio"].where(df["plr_ratio"] > 0))

    return df


def prepare_analysis_df(
    analysis_csv: Path | None = None,
    saliva_path: Path | None = None,
) -> pd.DataFrame:
    """Load the paired ratios, attach saliva periods, add circadian regressors."""
    analysis_csv = analysis_csv or DEFAULT_ANALYSIS_CSV
    if not analysis_csv.exists():
        raise FileNotFoundError(
            f"Paired ratios not found at {analysis_csv}. "
            "Run `python -m civibeplr.si_derive_protocol_plr` first."
        )

    ratios = pd.read_csv(analysis_csv)
    print(f"Loaded {len(ratios)} paired ratios from {analysis_csv.name}")

    saliva = load_saliva_results(saliva_path)
    merged = merge_pupil_saliva(ratios, saliva)

    before = len(merged)
    merged = merged.dropna(subset=["saliva_tau_hours"])
    if len(merged) < before:
        print(f"  Dropped {before - len(merged)} rows without a saliva tau estimate")

    df = add_circadian_regressors(merged)

    print(
        f"\nAnalysis dataframe: {len(df)} rows  |  "
        f"{df[ANALYSIS_PARTICIPANT_COL].nunique()} participants  |  "
        f"{df[PUPIL_CONDITION_COL].nunique()} conditions"
    )
    return df


def describe_outcome(df: pd.DataFrame, outcome_col: str) -> pd.DataFrame:
    """Per-condition descriptives for numerator, denominator and ratio."""
    summary = (
        df.groupby(PUPIL_CONDITION_COL)
        .agg(
            n_obs=(outcome_col, "count"),
            n_participants=(ANALYSIS_PARTICIPANT_COL, "nunique"),
            numerator_mean_mm=("mean_constriction_mm", "mean"),
            denominator_mean_mm=("flux_constriction_mm", "mean"),
            denominator_min_mm=("flux_constriction_mm", "min"),
            ratio_mean=(outcome_col, "mean"),
            ratio_sd=(outcome_col, "std"),
            ratio_median=(outcome_col, "median"),
            ratio_min=(outcome_col, "min"),
            ratio_max=(outcome_col, "max"),
            ratio_skew=(outcome_col, "skew"),
        )
        .reset_index()
    )
    summary["ratio_iqr"] = (
        df.groupby(PUPIL_CONDITION_COL)[outcome_col].quantile(0.75).values
        - df.groupby(PUPIL_CONDITION_COL)[outcome_col].quantile(0.25).values
    )
    return summary


def main(
    analysis_csv: Path | None = None,
    saliva_path: Path | None = None,
    outcome_col: str = "plr_ratio",
    force: bool = False,
    dry_run: bool = False,
    smc_draws: int = DEFAULT_SMC_DRAWS,
    random_seed: int = DEFAULT_SEED,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Fit M0/M1 per non-FLUX condition on the protocol ratio outcome."""
    if outcome_col not in OUTCOME_CHOICES:
        raise ValueError(f"outcome_col must be one of {OUTCOME_CHOICES}")

    output_dir = (output_dir or DEFAULT_OUTPUT_DIR) / outcome_col
    idata_dir = DEFAULT_IDATA_DIR / outcome_col

    df = prepare_analysis_df(analysis_csv, saliva_path)

    conditions = sorted(df[PUPIL_CONDITION_COL].dropna().unique())
    unexpected = set(conditions) - set(NON_FLUX_CONDITIONS)
    if unexpected:
        raise AssertionError(f"Unexpected conditions to model: {sorted(unexpected)}")
    if "flux" in conditions:
        raise AssertionError("FLUX must not be modelled: FLUX/FLUX is 1 by construction.")

    print("\n── Outcome descriptives ──")
    descriptives = describe_outcome(df, outcome_col)
    print(descriptives.round(3).to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    descriptives.to_csv(output_dir / f"{outcome_col}_descriptives.csv", index=False)

    if dry_run:
        models = {}
        print(f"\nDry run: building models for {len(conditions)} conditions...")
        for cond in conditions:
            df_cond = df[df[PUPIL_CONDITION_COL] == cond].dropna(subset=[outcome_col])
            print(
                f"  [{cond}]  n(Observations)={len(df_cond)}  "
                f"n(Participants)={df_cond[ANALYSIS_PARTICIPANT_COL].nunique()}"
            )
            models[cond] = build_models(df_cond, outcome_col=outcome_col)
        return df, models

    all_results: dict[str, dict] = {}
    for cond in conditions:
        df_cond = (
            df[df[PUPIL_CONDITION_COL] == cond].dropna(subset=[outcome_col]).copy()
        )
        print(f"\n{'=' * 70}\n[{cond}]  n={len(df_cond)}  "
              f"participants={df_cond[ANALYSIS_PARTICIPANT_COL].nunique()}\n{'=' * 70}")

        m0, m1 = build_models(df_cond, outcome_col=outcome_col)

        idata_m0, idata_m1 = (None, None) if force else load_idata(cond, idata_dir)
        if idata_m0 is None or idata_m1 is None:
            idata_m0, idata_m1 = fit_condition(
                m0, m1, smc_draws=smc_draws, random_seed=random_seed
            )
            save_idata(m1, idata_m0, idata_m1, cond,
                       outcome_col=outcome_col, idata_dir=idata_dir)

        all_results[cond] = extract_results(
            m0, m1, idata_m0, idata_m1, df_cond, outcome_col=outcome_col
        )

    save_results(all_results, output_dir=output_dir, parameter_col=outcome_col)
    return df, all_results


def inspect(outcome_col: str = "plr_ratio") -> pd.DataFrame:
    """Per-condition distribution of the outcome before modelling.

    Supplementary pipeline step 4b: the primary pipeline's
    ``inspect_parameter_distribution`` applied to the ratio frame.  The figure
    lands beside the primary ones as ``fig_parameter_distribution_<outcome>``.
    """
    if outcome_col not in OUTCOME_CHOICES:
        raise ValueError(f"outcome_col must be one of {OUTCOME_CHOICES}")
    return inspect_parameter_distribution(prepare_analysis_df(), parameter_col=outcome_col)


def diagnose(outcome_col: str = "plr_ratio") -> pd.DataFrame | None:
    """Run the primary pipeline's circadian diagnostics on the ratio outcome.

    Supplementary pipeline step 5b.  Returns None, with a message, when no
    fitted models exist yet for *outcome_col*.
    """
    if outcome_col not in OUTCOME_CHOICES:
        raise ValueError(f"outcome_col must be one of {OUTCOME_CHOICES}")

    idata_dir = DEFAULT_IDATA_DIR / outcome_col
    if not any(idata_dir.glob("*_M1.nc")):
        print(f"No fitted models in {idata_dir}; run step 5 with fitting first.")
        return None

    return run_diagnostics(parameter_col=outcome_col, idata_dir=idata_dir)


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Circadian models for the registered FLUX-normalised outcome."
    )
    parser.add_argument("--analysis-csv", type=Path, default=None)
    parser.add_argument("--saliva-path", type=Path, default=None)
    parser.add_argument("--outcome", choices=OUTCOME_CHOICES, default="plr_ratio",
                        help="plr_ratio is the registered outcome; log_plr_ratio is "
                             "a prespecified sensitivity analysis.")
    parser.add_argument("--fit", action="store_true",
                        help="Run SMC sampling. Without it, only a dry run is performed.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run SMC even when saved InferenceData exists.")
    parser.add_argument("--smc-draws", type=int, default=DEFAULT_SMC_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    main(
        analysis_csv=args.analysis_csv,
        saliva_path=args.saliva_path,
        outcome_col=args.outcome,
        force=args.force,
        dry_run=not args.fit,
        smc_draws=args.smc_draws,
        random_seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    _cli()
