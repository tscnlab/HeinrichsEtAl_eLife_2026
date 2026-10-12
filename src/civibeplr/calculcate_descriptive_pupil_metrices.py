"""Analyze aggregated pupil data - descriptive pupillary metrics.

Trial phases (pre-stim, stim, post-stim) can be defined either by stimulus
presentation labels or by time-based analysis windows. When use_analysis_window=True
(default), the stim/poststim boundary is set to DEFAULT_T_STIM_END + DEFAULT_ADD_POSTSTIM_SEC
from analysis_io.py. Setting DEFAULT_ADD_POSTSTIM_SEC = 0 aligns the analysis window
exactly with the stimulus window."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from civibeplr.data_io import (DEFAULT_BLOCK_COL, DEFAULT_CONDITION_COL,
                               DEFAULT_PARTICIPANT_COL, DEFAULT_TRIALPHASE_COL,
                               DEFAULT_TIME_COL, load_aggregated_data)
from civibeplr.analysis_io import (DEFAULT_T_STIM_START,
                                   DEFAULT_T_STIM_END,
                                   DEFAULT_ADD_POSTSTIM_SEC)
from civibeplr.utils_cosinor import format_number
from civibeplr.latex_table_creation import save_selected_pupil_metrics_latex

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = DEFAULT_PROJECT_DIR / "results"

default_t_resp_start = DEFAULT_T_STIM_START
default_t_resp_end = DEFAULT_T_STIM_END + DEFAULT_ADD_POSTSTIM_SEC


def save_pupil_metrics_metadata(pupil_metrics, output_path):
    """Save the pupil metrics metadata to a CSV file."""

    # Check if metrics exists
    if pupil_metrics is None or pupil_metrics.empty:
        print("No pupil metrics to save.")
        return

    metrics_metadata = [
        {
            "metric": "prestim_mean",
            "description": "Mean pupil size during pre-stimulus window",
            "unit": "mm",
        },
        {
            "metric": "stim_mean",
            "description": "Mean pupil size during stimulus window",
            "unit": "mm",
        },
        {
            "metric": "poststim_mean",
            "description": "Mean pupil size during post-stimulus window",
            "unit": "mm",
        },
        {
            "metric": "prestim_std",
            "description": "Std dev of pupil size during pre-stimulus window",
            "unit": "mm",
        },
        {
            "metric": "stim_std",
            "description": "Std dev of pupil size during stimulus window",
            "unit": "mm",
        },
        {
            "metric": "poststim_std",
            "description": "Std dev of pupil size during post-stimulus window",
            "unit": "mm",
        },
        {
            "metric": "prestim_mean_pct",
            "description": "Mean % change from baseline during pre-stimulus window",
            "unit": "%",
        },
        {
            "metric": "stim_mean_pct",
            "description": "Mean % change from baseline during stimulus window",
            "unit": "%",
        },
        {
            "metric": "poststim_mean_pct",
            "description": "Mean % change from baseline during post-stimulus window",
            "unit": "%",
        },
        {
            "metric": "prestim_std_pct",
            "description": "Std dev of % change from baseline during pre-stimulus window",
            "unit": "%",
        },
        {
            "metric": "stim_std_pct",
            "description": "Std dev of % change from baseline during stimulus window",
            "unit": "%",
        },
        {
            "metric": "poststim_std_pct",
            "description": "Std dev of % change from baseline during post-stimulus window",
            "unit": "%",
        },
        {
            "metric": "stim_minus_prestim",
            "description": "Difference in mean pupil size (stim - pre-stim)",
            "unit": "mm",
        },
        {
            "metric": "amplitude_stim",
            "description": "Minimum rel_size_mm within 1s of stim onset (max constriction)",
            "unit": "mm",
        },
        {
            "metric": "auc_stim",
            "description": "Area under curve (AUC) of rel_size_mm during stimulus window",
            "unit": "mm·s",
        },
        {
            "metric": "auc_poststim_early",
            "description": "AUC of rel_size_mm during early post-stimulus (0-5s after offset)",
            "unit": "mm·s",
        },
        {
            "metric": "auc_poststim_late",
            "description": "AUC of rel_size_mm during late post-stimulus (>6s after offset)",
            "unit": "mm·s",
        },
    ]
    df = pd.DataFrame(metrics_metadata)
    df.to_csv(output_path, index=False)
    return df


def calculate_pupil_metrics(
    df: pd.DataFrame,
    grouping_cols: list[str] | None = None,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    time_col: str = DEFAULT_TIME_COL,
    trialphase_col: str = DEFAULT_TRIALPHASE_COL,
    use_analysis_window: float = True,
) -> pd.DataFrame:
    """Calculate pupillary metrics for each participant × condition × block.

    Computes summary metrics from specific outcome columns:
    - Mean and std from pupil_mean (absolute size) for each trial phase
    - Mean and std from change_size_pct (percentage change) for each trial phase
    - Difference between stimulus and pre-stimulus means
    - Amplitude: minimum rel_size_mm within 1 second of stim onset (maximum constriction)
    - AUC from rel_size_mm (baseline-corrected) during the analysis window (stim/poststim boundary set by DEFAULT_ADD_POSTSTIM_SEC) and post-stimulus phases (split into early 0-6s and late >6s after stimulus offset)

    Args:
        df: Aggregated pupil data with phase information.
        grouping_cols: List of columns to group by. If None, uses [participant, condition, block].
        participant_col: Column name for participant ID.
        condition_col: Column name for condition/trial type.
        block_col: Column name for block.
        time_col: Column name for time.
        trialphase_col: Column name for trial phase (stim, pre-stim, post-stim).
        use_analysis_window: If True, defines trial phases based on time-based windows
            (stim/poststim boundary = DEFAULT_T_STIM_END + DEFAULT_ADD_POSTSTIM_SEC)
            rather than phase labels in the data.

    Returns:
        DataFrame with one row per participant x condition x block and columns for each metric.
    """
    if grouping_cols is None:
        grouping_cols = [participant_col, condition_col, block_col]

    # Validate required columns
    required_cols = grouping_cols + [time_col, trialphase_col, "pupil_mean", "change_size_pct", "rel_size_mm"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    metrics_list = []

    for group_key, group_data in df.groupby(grouping_cols, dropna=False):
        # Create dict for this group's metrics
        metrics = {}

        # Store grouping values
        for i, col in enumerate(grouping_cols):
            metrics[col] = group_key[i] if isinstance(group_key, tuple) else group_key

        # Sort by time for proper integration
        group_data = group_data.sort_values(time_col)

        if use_analysis_window:
            # print(f"\nCalculating metrics based on time-based analysis window, with stimulus window defined as {DEFAULT_T_STIM_START} to {DEFAULT_T_STIM_END} seconds (plus {DEFAULT_ADD_POSTSTIM_SEC} sec post-stimulus).")
            # Create time-based masks for each phase
            t = group_data[time_col].values.astype(float)
            # Define based on analysis windows rather than phase labels
            prestim_mask = t < DEFAULT_T_STIM_START
            stim_mask = (t >= DEFAULT_T_STIM_START) & (t <= DEFAULT_T_STIM_END + DEFAULT_ADD_POSTSTIM_SEC)
            poststim_mask = t > (DEFAULT_T_STIM_END + DEFAULT_ADD_POSTSTIM_SEC)

            # Define based on phase labels (if available and consistent)
            prestim_data = group_data[prestim_mask]
            stim_data = group_data[stim_mask]
            poststim_data = group_data[poststim_mask]
        else:
            print(f"\nCalculating metrics based on trial phase labels in column '{trialphase_col}' (pre-stim, stim, post-stim).")
            trial_phase = group_data[trialphase_col].astype(str).str.lower()
            prestim_data = group_data[trial_phase == "pre-stim"]
            stim_data = group_data[trial_phase == "stim"]
            poststim_data = group_data[trial_phase == "post-stim"]

        # Calculate means and std for pupil_mean (absolute size)
        metrics["prestim_mean"] = prestim_data["pupil_mean"].mean() if not prestim_data.empty else np.nan
        metrics["stim_mean"] = stim_data["pupil_mean"].mean() if not stim_data.empty else np.nan
        metrics["poststim_mean"] = poststim_data["pupil_mean"].mean() if not poststim_data.empty else np.nan

        metrics["prestim_std"] = prestim_data["pupil_mean"].std() if not prestim_data.empty else np.nan
        metrics["stim_std"] = stim_data["pupil_mean"].std() if not stim_data.empty else np.nan
        metrics["poststim_std"] = poststim_data["pupil_mean"].std() if not poststim_data.empty else np.nan

        # Calculate means and std for change_size_pct (percentage change)
        metrics["prestim_mean_pct"] = prestim_data["change_size_pct"].mean() if not prestim_data.empty else np.nan
        metrics["stim_mean_pct"] = stim_data["change_size_pct"].mean() if not stim_data.empty else np.nan
        metrics["poststim_mean_pct"] = poststim_data["change_size_pct"].mean() if not poststim_data.empty else np.nan

        metrics["prestim_std_pct"] = prestim_data["change_size_pct"].std() if not prestim_data.empty else np.nan
        metrics["stim_std_pct"] = stim_data["change_size_pct"].std() if not stim_data.empty else np.nan
        metrics["poststim_std_pct"] = poststim_data["change_size_pct"].std() if not poststim_data.empty else np.nan

        # Calculate difference between stim and pre-stim (from pupil_mean)
        metrics["stim_minus_prestim"] = metrics["stim_mean"] - metrics["prestim_mean"]

        # Calculate amplitude: minimum rel_size_mm within 1 second of stim onset
        if not stim_data.empty:
            # Get stimulus onset time (min time in stim phase)
            stim_onset_time = stim_data[time_col].min()
            # Filter for data within 1 second of onset
            early_stim = stim_data[stim_data[time_col] <= stim_onset_time + 1.0]
            if not early_stim.empty:
                # Minimum value = maximum constriction (most negative value relative to baseline)
                metrics["amplitude_stim"] = early_stim["rel_size_mm"].min()
            else:
                metrics["amplitude_stim"] = np.nan
        else:
            metrics["amplitude_stim"] = np.nan

        # Calculate AUC from rel_size_mm (baseline-corrected) using trapezoidal rule
        if not stim_data.empty and len(stim_data) > 1:
            metrics["auc_stim"] = np.trapezoid(stim_data["rel_size_mm"], stim_data[time_col])
        else:
            metrics["auc_stim"] = np.nan

        # Calculate post-stimulus AUC with time windows (from rel_size_mm)
        if not stim_data.empty and not poststim_data.empty:
            # Get stimulus offset time (max time in stim phase)
            stim_offset_time = stim_data[time_col].max()

            # Calculate time relative to stimulus offset
            poststim_data = poststim_data.copy()
            poststim_data["time_rel_offset"] = poststim_data[time_col] - stim_offset_time

            # Early post-stim: 0-5 seconds after offset
            early_poststim = poststim_data[
                (poststim_data["time_rel_offset"] >= default_t_resp_start)
                & (poststim_data["time_rel_offset"] <= default_t_resp_end)
            ]
            if not early_poststim.empty and len(early_poststim) > 1:
                metrics["auc_poststim_early"] = np.trapezoid(
                    early_poststim["rel_size_mm"], 
                    early_poststim[time_col]
                )
            else:
                metrics["auc_poststim_early"] = np.nan

            # Late post-stim: >6 seconds after offset
            late_poststim = poststim_data[poststim_data["time_rel_offset"] > 6]
            if not late_poststim.empty and len(late_poststim) > 1:
                metrics["auc_poststim_late"] = np.trapezoid(
                    late_poststim["rel_size_mm"], 
                    late_poststim[time_col]
                )
            else:
                metrics["auc_poststim_late"] = np.nan
        else:
            metrics["auc_poststim_early"] = np.nan
            metrics["auc_poststim_late"] = np.nan

        metrics_list.append(metrics)

    result = pd.DataFrame(metrics_list)
    return result


def main(
    data_path: Path | None = None,
    output: Path | None = None,
) -> pd.DataFrame:
    """Calculate and save pupil metrics.

    Parameters
    ----------
    data_path : Path or None
        Path to aggregated data CSV.  Defaults to data/aggregated/data.csv.
    output : Path or None
        Output path for metrics CSV.  Defaults to results/descriptives/pupil_metrics.csv.

    Returns
    -------
    pd.DataFrame
        Metrics table (one row per participant × condition × block).
    """
    df = load_aggregated_data(data_path)

    print(f"\nCalculating pupil metrics...")
    metrics = calculate_pupil_metrics(df)

    print(f"\nCalculated metrics for {len(metrics)} participant × condition × block combinations.")
    print("\nMetrics summary:")
    print(metrics.describe())

    output_path = output or DEFAULT_RESULTS_DIR / "descriptives" / "pupil_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # metrics.to_csv(output_path, index=False)
    # print(f"\nSaved metrics to: {output_path}")

    metadata_path = output_path.parent / "pupil_metrics_metadata.csv"
    save_pupil_metrics_metadata(pupil_metrics=metrics, output_path=metadata_path)
    print(f"Saved metrics metadata to: {metadata_path}")

    metrics_formatted = metrics.copy()
    numeric_cols = metrics_formatted.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        metrics_formatted[col] = metrics_formatted[col].apply(format_number)

    latex_path = output_path.parent / f"{output_path.stem}.tex"
    latex_table = metrics_formatted.to_latex(index=False, escape=False)
    latex_path.write_text(latex_table)
    print(f"Saved LaTeX table to: {latex_path}")

    manuscript_path = output_path.parent / "tab_pupil_descriptive_metrics_selected.tex"
    save_selected_pupil_metrics_latex(metrics=metrics, output_path=manuscript_path)
    print(f"Saved manuscript metrics table to: {manuscript_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze aggregated pupil data with metrics and sine wave fitting."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to aggregated data CSV (default: data/aggregated/data.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for metrics CSV (default: results/descriptives/pupil_metrics.csv).",
    )
    args = parser.parse_args()
    main(data_path=args.data_path, output=args.output)
