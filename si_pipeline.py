"""Supplementary pipeline for the registered FLUX-normalised outcome.

Deliberately separate from ``pipeline.py``: it is not registered in that
file's ``ALL_STEPS``, so running the primary pipeline never touches it.

Step IDs follow ``pipeline.py`` so the same number means the same stage in
both: Step 3 is where the per-block outcome is produced, 4b inspects its
distribution, Step 5 fits the circadian models, 5b diagnoses them and 5c plots
them.  Steps 1-2 have no counterpart (the endpoint is derived straight from the
per-participant files) and Step 4, the saliva cosinor, is read from the primary
pipeline's output rather than re-run.  Step 6 has no primary counterpart.

Outputs follow the primary layout, keyed by outcome name exactly as
``amplitude`` is: Steps 4b-5c write to ``circadian_cosinor*/plr_ratio/`` and
``circadian_cosinor*/log_plr_ratio/``.  Those names cannot collide with a
primary outcome.  Step 3, 3b and 6 outputs, which have no primary counterpart,
go to ``plr_ratio/`` folders under ``data/derivatives`` and ``results``.

Can be run as a script:
    python si_pipeline.py                  # all steps, models as dry run
    python si_pipeline.py --fit            # all steps, with SMC sampling
    python si_pipeline.py --steps 3 3b
    python si_pipeline.py --steps 5 5b --fit --outcomes log_plr_ratio

Or imported and called from a Jupyter notebook:
    from si_pipeline import run_pipeline
    run_pipeline(steps=["5", "5b", "5c"], outcomes=["plr_ratio"], fit=True)

The raw ratio is skewed (Shapiro p < 0.05 in every condition), so Step 5b is
not optional: the Gaussian likelihood has to be checked rather than assumed.
``log_plr_ratio`` is the prespecified sensitivity analysis.

Requires the saliva cosinor (pipeline Step 4) to have been run, since the
circadian regressors use the participant-specific periods it estimates.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from civibeplr.analyse_circadian_pupil_parameters import DEFAULT_SMC_DRAWS
from civibeplr.si_analyse_plr_ratio import (
    DEFAULT_SEED,
    OUTCOME_CHOICES,
    diagnose as diagnose_protocol_plr,
    inspect as inspect_protocol_plr,
    main as analyse_protocol_plr,
)
from civibeplr.si_derive_protocol_plr import derive_plr_ratio_endpoint
from civibeplr.si_sensitivity_protocol_plr import main as si_sensitivity_protocol_plr
from civibeplr.si_visualise_protocol_plr import circadian_figures, endpoint_figures


ALL_STEPS = ["3", "3b", "4b", "5", "5b", "5c", "6"]

STEP_LABELS = {
    "3": "Derive protocol endpoint (replaces primary Steps 1-3)",
    "3b": "Visualise protocol endpoint",
    "4b": "Inspect outcome distributions",
    "5": "Analyse circadian protocol ratio",
    "5b": "Diagnose circadian models",
    "5c": "Visualise circadian fits",
    "6": "Sensitivity analyses",
}


def _pipeline_header(step: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  STEP {step}: {STEP_LABELS[step]}")
    print(f"{'=' * 65}")


def _outcome_header(outcome: str, n_outcomes: int) -> None:
    if n_outcomes > 1:
        print(f"\n{'='*8} OUTCOME: {outcome.upper()} {'='*8}")


def run_pipeline(
    steps: Sequence[str] = ALL_STEPS,
    outcomes: Sequence[str] = OUTCOME_CHOICES,
    fit: bool = False,
    force: bool = False,
    smc_draws: int = DEFAULT_SMC_DRAWS,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Run the supplementary protocol-outcome pipeline.

    Parameters
    ----------
    steps : sequence of str
        Steps to run, in any order; always executed in pipeline order.
        Choices: "3", "3b", "4b", "5", "5b", "5c", "6".
    outcomes : sequence of str
        Outcomes for Steps 4b, 5, 5b, 5c, in any order; always executed in
        OUTCOME_CHOICES order.  ``plr_ratio`` is the registered outcome,
        ``log_plr_ratio`` the prespecified sensitivity analysis.  Step 6 always
        uses ``plr_ratio``.
    fit : bool
        If True, run SMC sampling in Step 5 and the sensitivity refits in
        Step 6.  If False, Step 5 is a dry run and Step 6 is skipped.
    force : bool
        If True, re-run SMC sampling in Step 5 even when saved InferenceData
        already exists.
    smc_draws : int
        Number of SMC draws.  Do not lower this: the SMC estimate of log Z is
        biased away from zero with too few particles.
    seed : int
        Random seed for SMC sampling.

    Returns
    -------
    dict
        Intermediate outputs keyed by step, with "<step>/<outcome>" for the
        per-outcome steps.
    """
    unknown = set(steps) - set(ALL_STEPS)
    if unknown:
        raise ValueError(f"Unknown step(s): {sorted(unknown)}. Choose from {ALL_STEPS}.")

    ordered = [s for s in ALL_STEPS if s in steps]
    ordered_outcomes = [o for o in OUTCOME_CHOICES if o in outcomes]
    results: dict = {}

    # No Step 0: demographics do not depend on the outcome and are already
    # written by the primary pipeline.

    # Differs from primary Step 1-3: instead of fitting the per-block cosinor to aggregated data (primary Steps 1-3), the protocol ratio is derived directly from the per-participant files, so Steps 1-2 are not needed.
    if "3" in ordered:
        _pipeline_header("3")
        results["3"] = derive_plr_ratio_endpoint()

    # Differs from primary Step 3b: plots the protocol endpoint, not pupil decomposition and pupil cosinor fits.
    if "3b" in ordered:
        _pipeline_header("3b")
        results["3b"] = endpoint_figures()

    # No Step 3c: there is no per-block cosinor to diagnose.
    # No Step 4: the saliva cosinor is read from the primary pipeline's output
    # rather than re-fitted, so both analyses use the same circadian periods.

    # Steps 4b-5c loop over `outcomes` (plr_ratio, log_plr_ratio) where the
    # primary pipeline loops over `parameters`; otherwise they mirror it.
    if "4b" in ordered:
        _pipeline_header("4b")
        for outcome in ordered_outcomes:
            _outcome_header(outcome, len(ordered_outcomes))
            results[f"4b/{outcome}"] = inspect_protocol_plr(outcome_col=outcome)

    # Differs from primary Step 5: sampling is opt-in via `fit` (dry run by
    # default), whereas the primary pipeline always fits or loads, so a full
    # run here cannot start a long SMC job by accident.
    if "5" in ordered:
        _pipeline_header("5")
        if not fit:
            print("Dry run; pass --fit to sample.")
        for outcome in ordered_outcomes:
            _outcome_header(outcome, len(ordered_outcomes))
            results[f"5/{outcome}"] = analyse_protocol_plr(
                outcome_col=outcome,
                force=force,
                dry_run=not fit,
                smc_draws=smc_draws,
                random_seed=seed,
            )

    if "5b" in ordered:
        _pipeline_header("5b")
        for outcome in ordered_outcomes:
            _outcome_header(outcome, len(ordered_outcomes))
            results[f"5b/{outcome}"] = diagnose_protocol_plr(outcome_col=outcome)

    if "5c" in ordered:
        _pipeline_header("5c")
        for outcome in ordered_outcomes:
            _outcome_header(outcome, len(ordered_outcomes))
            results[f"5c/{outcome}"] = circadian_figures(outcome_col=outcome)

    # No primary counterpart: sensitivity analyses specific to the protocol
    # outcome. Refits models, so it only runs with `fit`.
    if "6" in ordered:
        _pipeline_header("6")
        if fit:
            results["6"] = si_sensitivity_protocol_plr(
                outcome_col="plr_ratio", smc_draws=smc_draws, seed=seed
            )
        else:
            print("Skipped: the sensitivity analyses refit models; pass --fit to run them.")

    print(f"\n{'=' * 65}")
    print("  Supplementary pipeline complete.")
    print(f"{'=' * 65}\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the supplementary protocol-outcome pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps:\n" + "\n".join(f"  {k}: {v}" for k, v in STEP_LABELS.items()),
    )
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS, default=ALL_STEPS,
                        metavar="STEP",
                        help=f"Steps to run (default: all). Choices: {', '.join(ALL_STEPS)}.")
    parser.add_argument("--outcomes", nargs="+", choices=OUTCOME_CHOICES,
                        default=OUTCOME_CHOICES, metavar="OUTCOME",
                        help="Outcomes for Steps 4b, 5, 5b, 5c (default: all). plr_ratio is "
                             "the registered outcome; log_plr_ratio is a prespecified "
                             "sensitivity analysis.")
    parser.add_argument("--fit", action="store_true",
                        help="Run SMC sampling. Without it Step 5 is a dry run and "
                             "Step 6 is skipped.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run SMC even when saved InferenceData exists.")
    parser.add_argument("--smc-draws", type=int, default=DEFAULT_SMC_DRAWS,
                        help="Do not lower this; see CLAUDE.md on log-Z bias.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    run_pipeline(
        steps=args.steps,
        outcomes=args.outcomes,
        fit=args.fit,
        force=args.force,
        smc_draws=args.smc_draws,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
