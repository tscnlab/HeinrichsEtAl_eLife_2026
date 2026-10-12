"""End-to-end analysis pipeline for civibeplr.

Can be run as a script:
    python pipeline.py
    python pipeline.py --steps 1 2 3 3b
    python pipeline.py --fit --seed 1072
    python pipeline.py --steps 5c --parameters amplitude tonic_min

Or imported and called from a Jupyter notebook:
    from pipeline import run_pipeline
    run_pipeline()
    run_pipeline(steps=["1", "2", "3", "3b"])
    run_pipeline(steps=["5c"], parameters=["amplitude", "tonic_min"])
"""

from __future__ import annotations

import argparse
from typing import Sequence

import pandas as pd

from civibeplr.aggregate_pupil_traces import main_all as aggregate
from civibeplr.analysis_io import PARAMETER_CHOICES
from civibeplr.get_participant_info import main as get_participant_info
from civibeplr.calculcate_descriptive_pupil_metrices import main as calculate_metrics
from civibeplr.data_io import load_aggregated_data
from civibeplr.diagnose_circadian_models import run_diagnostics as diagnose_circadian
from civibeplr.diagnose_pupil_models import (
    run_diagnostics as diagnose_pupil,
    DEFAULT_RESULTS_CSV as PUPIL_DIAG_RESULTS_CSV,
    DEFAULT_FIG_DIR as PUPIL_DIAG_FIG_DIR,
)
from civibeplr.fit_pupil_trace_2step_cosinor_model import main as fit_cosinor
from civibeplr.fit_saliva_cosinor_model import main as fit_saliva
from civibeplr.analyse_circadian_pupil_parameters import (
    main as analyse_circadian,
    DEFAULT_SMC_DRAWS,
)
from civibeplr.inspect_circadian_model import main as inspect_model
from civibeplr.visualise_circadian_parameter_cosinor import main as visualise_circadian
from civibeplr.visualise_pupil_cosinor import parameter_plots, decomposition_plots


ALL_STEPS = [
    "0", "1", "2", "3", "3b", "3c", "4", "4b", "5", "5b", "5c"
    ]

STEP_LABELS = {
    "0": "Extract participant demographics",
    "1": "Aggregate pupil traces",
    "2": "Calculate pupil metrics",
    "3": "Fit 2-step cosinor model (pupil)",
    "3b": "Visualise pupil cosinor fits",
    "3c": "Diagnose pupil cosinor models",
    "4": "Fit saliva cosinor model",
    "4b": "Inspect outcome distributions",
    "5": "Analyse circadian pupil parameters",
    "5b": "Diagnose circadian models",
    "5c": "Visualise circadian cosinor fits",
}


def _pipeline_header(step: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  STEP {step}: {STEP_LABELS[step]}")
    print(f"{'=' * 65}")


def run_pipeline(
    steps: Sequence[str] = ALL_STEPS,
    force: bool = False,
    dry_run: bool = False,
    smc_draws: int = DEFAULT_SMC_DRAWS,
    seed: int = 1072,
    plot_individual: bool = False,
    parameters: Sequence[str] = PARAMETER_CHOICES,
) -> dict:
    """Run the civibeplr analysis pipeline.

    Parameters
    ----------
    steps : sequence of str
        Pipeline steps to run, in any order; always executed in pipeline order.
        Choices: "1", "2", "3", "3b", "3c", "4", "4b", "5", "5b", "5c".
    force : bool
        If True, re-run SMC sampling in Step 5 even when saved InferenceData
        already exists.  If False (default), load existing files and only fit
        conditions whose files are missing.
    dry_run : bool
        If True, build and compile Step 5 models only without fitting or loading.
    smc_draws : int
        Number of SMC draws for Step 5 (default DEFAULT_SMC_DRAWS). Do not lower
        this: the SMC estimate of log Z is biased away from zero with too few
        particles, which inflates the apparent evidence for the null model.
    seed : int
        Random seed for Step 5 SMC sampling.
    plot_individual : bool
        If True, generate per-participant decomposition plots in Step 3b (slow).
    parameters : sequence of str
        Outcome parameters to run for Steps 4b, 5, 5b, 5c, in any order;
        always executed in PARAMETER_CHOICES order.
        Step 5c writes the combined figure
        `results/figures/circadian_cosinor/<parameter>/fig_fitted_vs_empirical_all_conditions.png`
        for each selected parameter.
        Choices: see PARAMETER_CHOICES in analysis_io.

    Returns
    -------
    dict
        Intermediate DataFrames keyed by step ("1", "2", "3", "4", "5").
    """
    ordered_params = [p for p in PARAMETER_CHOICES if p in parameters]
    ordered = [s for s in ALL_STEPS if s in steps]
    results: dict[str, pd.DataFrame] = {}

    if "0" in ordered:
        _pipeline_header("0")
        results["0"] = get_participant_info()

    if "1" in ordered:
        _pipeline_header("1")
        results["1"] = aggregate(plot_by_participant=plot_individual)

    if "2" in ordered:
        _pipeline_header("2")
        results["2"] = calculate_metrics()

    if "3" in ordered:
        _pipeline_header("3")
        df = results.get("1") if results.get("1") is not None else load_aggregated_data()
        results["3"] = fit_cosinor(df)

    if "3b" in ordered:
        _pipeline_header("3b")
        parameter_plots()
        decomposition_plots(
            results_df=results.get("3"),
            plot_individual_participants=plot_individual,
        )

    if "3c" in ordered:
        _pipeline_header("3c")
        diagnose_pupil(results_csv=PUPIL_DIAG_RESULTS_CSV, fig_dir=PUPIL_DIAG_FIG_DIR)

    if "4" in ordered:
        _pipeline_header("4")
        results["4"], _ = fit_saliva()

    if "4b" in ordered:
        _pipeline_header("4b")
        for parameter in ordered_params:
            if len(ordered_params) > 1:
                print(f"\n{'='*8} OUTCOME PARAMETER: {parameter.upper()} {'='*8}")
            inspect_model(outcome_col=parameter)

    if "5" in ordered:
        _pipeline_header("5")
        for parameter in ordered_params:
            if len(ordered_params) > 1:
                print(
                    f"{'='*30}\n{'='*8} OUTCOME PARAMETER: {parameter.upper()} {'='*8}\n{'='*30}"
                )
            results[f"5/{parameter}"], _ = analyse_circadian(
                force=force,
                dry_run=dry_run,
                smc_draws=smc_draws,
                random_seed=seed,
                outcome_parameter=parameter,
            )

    if "5b" in ordered:
        _pipeline_header("5b")
        for parameter in ordered_params:
            if len(ordered_params) > 1:
                print(f"\n{'='*8} OUTCOME PARAMETER: {parameter.upper()} {'='*8}")
            results[f"5b/{parameter}"] = diagnose_circadian(parameter_col=parameter)

    if "5c" in ordered:
        _pipeline_header("5c")
        for parameter in ordered_params:
            if len(ordered_params) > 1:
                print(f"\n{'='*8} OUTCOME PARAMETER: {parameter.upper()} {'='*8}")
            visualise_circadian(outcome_col=parameter)

    print(f"\n{'=' * 65}")
    print("  Pipeline complete.")
    print(f"{'=' * 65}\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the civibeplr analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Steps:\n" + "\n".join(f"  {k}: {v}" for k, v in STEP_LABELS.items()),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        metavar="STEP",
        help=f"Steps to run (default: all). Choices: {', '.join(ALL_STEPS)}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run SMC sampling in Step 5 even when saved InferenceData exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and compile Step 5 models only, without fitting or loading.",
    )
    parser.add_argument(
        "--smc-draws",
        type=int,
        default=DEFAULT_SMC_DRAWS,
        help=f"Number of SMC draws for Step 5 (default: {DEFAULT_SMC_DRAWS}). "
             "Lowering this biases the Bayes factors towards the null model.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1072,
        help="Random seed for Step 5 SMC sampling (default: 1072).",
    )
    parser.add_argument(
        "--plot-individual",
        action="store_true",
        help="Generate per-participant decomposition plots in Step 3b (slow).",
    )
    parser.add_argument(
        "--parameters",
        nargs="+",
        choices=PARAMETER_CHOICES,
        default=PARAMETER_CHOICES,
        metavar="PARAMETER",
        help=f"Outcome parameters to run for Steps 4b, 5, 5b, 5c (default: all). Choices: {', '.join(PARAMETER_CHOICES)}.",
    )
    args = parser.parse_args()


    # Run the pipeline with the specified arguments
    run_pipeline(
        steps=args.steps,
        force=args.force,
        dry_run=False, #args.dry_run,
        smc_draws=args.smc_draws,
        seed=args.seed,
        plot_individual=False, #args.plot_individual,
        parameters=args.parameters,
    )


if __name__ == "__main__":
    main()
