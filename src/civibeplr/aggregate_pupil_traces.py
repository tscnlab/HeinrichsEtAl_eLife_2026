"""Aggregate participant pupil traces by condition and block."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from civibeplr.data_io import filter_min_blocks
from civibeplr.latex_table_creation import format_latex_variables
from civibeplr.vis_io import _in_jupyter, apply_font_style, TRIAL_TYPE_COLORS

apply_font_style()
from civibeplr.load_final_data import _iter_csvs, load_participant_csv

DEFAULT_PROJECT_DIR = (
    Path(__file__).resolve().parents[2]
)
DEFAULT_FINAL_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "retinawise_results" / "final"
)
DEFAULT_FIG_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "figures" 
)

DEFAULT_TRIAL_COUNT_TEX = DEFAULT_PROJECT_DIR / "results" / "tex" / "aggregated" / "vars_trial_counts.tex"

# Scheduled design: two runs per block, each of 25 trials (five per condition).
SCHEDULED_RUNS_PER_BLOCK = 2
TRIALS_PER_RUN = 25

DEFAULT_CONDITION_COL = "Trial type"
DEFAULT_BLOCK_COL = "Block"
DEFAULT_TEST_COL = "Test"
DEFAULT_TRIAL_NO_COL = "Trial no"
DEFAULT_TIME_COL = "Trial time Sec"
DEFAULT_EYE_COL = "Eye"
DEFAULT_LEFT_COL = "Left - Size Mm"
DEFAULT_RIGHT_COL = "Right - Size Mm"
DEFAULT_PHASE_COL = "Trial phase"
DEFAULT_GROUPING_COLS = [DEFAULT_CONDITION_COL, DEFAULT_BLOCK_COL, DEFAULT_TIME_COL, DEFAULT_PHASE_COL]


def _select_pupil_size(df: pd.DataFrame) -> pd.Series:
    """Select pupil size based on measured eye.
    
    Args:
        df: Input dataframe.
        
    Returns:
        Series with pupil size values.
    """
    required_cols = [DEFAULT_EYE_COL, DEFAULT_LEFT_COL, DEFAULT_RIGHT_COL]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: {col}")
    
    eye = df[DEFAULT_EYE_COL].astype(str).str.upper()
    left = df[DEFAULT_LEFT_COL]
    right = df[DEFAULT_RIGHT_COL]

    pupil = left.where(eye == "L", other=right.where(eye == "R"))
    if pupil.isna().all():
        raise ValueError(
            "Could not derive pupil size: check `Eye`, `Left - Size Mm`, `Right - Size Mm` columns."
        )
    return pupil


def aggregate_participant_traces(
    df: pd.DataFrame,
    grouping_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate pupil traces per condition and block.

    Groups over tests to compute mean and std at each time point. 

    Args:
        df: Participant data frame.
        grouping_cols: List of column names to group by. Defaults to [condition, block, time_sec].

    Returns:
        DataFrame with mean and std columns per time point, grouped by participant, condition, block, and time.
    """

    grouping_cols = grouping_cols or DEFAULT_GROUPING_COLS.copy()

    for col in grouping_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required grouping column: {col}")

    df = df.copy()
    df["pupil_size_mm"] = _select_pupil_size(df)

    group_cols = grouping_cols.copy()
    if "Participant id" in df.columns and "Participant id" not in group_cols:
        group_cols.insert(0, "Participant id")

    # Tests are intentionally not included in grouping; mean/std aggregate across tests.
    grouped = (
        df.groupby(group_cols, dropna=False)["pupil_size_mm"]
        .agg(pupil_mean="mean", pupil_std="std", n="count")
        .reset_index()
    )

    return grouped


def _stim_windows(
    df: pd.DataFrame,
    condition_col: str,
    block_col: str,
    time_col: str,
    trial_phase_col: str,
) -> dict[tuple, tuple[float, float]]:
    """Calculate stim windows from data containing phase information.
    Extracts start and end times of the 'stim' phase for each condition and 
    block to enable shading of stim periods in plots.
    
    Args:
        df: DataFrame with phase, condition, block, and time columns.
        condition_col: Column name for condition.
        block_col: Column name for block.
        time_col: Column name for time.
        trial_phase_col: Column name for trial phase.
        
    Returns:
        Dict mapping (condition, block) to (start_time, end_time) tuples.
    """
    if trial_phase_col not in df.columns:
        return {}

    stim_rows = df[df[trial_phase_col].astype(str).str.lower() == "stim"]
    if stim_rows.empty:
        return {}

    windows: dict[tuple, tuple[float, float]] = {}
    for (condition, block), group in stim_rows.groupby([condition_col, block_col], dropna=False):
        t_min = group[time_col].min()
        t_max = group[time_col].max()
        if pd.notna(t_min) and pd.notna(t_max):
            windows[(condition, block)] = (float(t_min), float(t_max))
    return windows


def normalise_to_baseline(
    df: pd.DataFrame,
    trial_phase_col: str = DEFAULT_PHASE_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    participant_col: str = "Participant id",
    baseline_method: str = "mean",
) -> pd.DataFrame:
    """Normalise pupil traces to pre-stim baseline.
    
    Calculates baseline from 'pre-stim' phase and adds normalized columns.
    
    Args:
        df: Aggregated pupil data with trial phase information.
        trial_phase_col: Column name for trial phase.
        condition_col: Column name for condition.
        block_col: Column name for block.
        participant_col: Column name for participant ID.
        baseline_method: Method to calculate baseline ('mean' or 'median').
        
    Returns:
        DataFrame with additional columns:
            - 'baseline_mm': Baseline pupil size
            - 'rel_size_mm': Relative size (difference from baseline)
            - 'change_size_pct': Percentage change from baseline
    """
    if trial_phase_col not in df.columns:
        raise KeyError(f"Missing trial phase column: {trial_phase_col}")
    if "pupil_mean" not in df.columns:
        raise KeyError("Missing 'pupil_mean' column in aggregated data")
    
    df = df.copy()
    
    # Identify grouping columns
    group_cols = []
    if participant_col in df.columns:
        group_cols.append(participant_col)
    group_cols.extend([condition_col, block_col])
    
    # Calculate baseline for each group
    prestim_mask = df[trial_phase_col].astype(str).str.lower() == "pre-stim"
    prestim_data = df[prestim_mask].copy()
    
    if baseline_method == "median":
        baselines = prestim_data.groupby(group_cols, dropna=False)["pupil_mean"].median()
    else:  # default to mean
        baselines = prestim_data.groupby(group_cols, dropna=False)["pupil_mean"].mean()
    
    baselines = baselines.rename("baseline_mm").reset_index()
    
    # Merge baseline back to original dataframe
    df = df.merge(baselines, on=group_cols, how="left")
    
    # Calculate normalized columns
    df["rel_size_mm"] = df["pupil_mean"] - df["baseline_mm"]
    df["change_size_pct"] = ((df["pupil_mean"] - df["baseline_mm"]) / df["baseline_mm"]) * 100
    
    return df


def depr_generate_ratio_to_reference(
    df: pd.DataFrame,
    target_col: str = "rel_size_mm",
    condition_col: str = DEFAULT_CONDITION_COL,
    grouping_cols: list[str] | None = None,
    reference_condition: str = "flux",
) -> pd.DataFrame:
    """Calculate ratio of pupil traces relative to a reference condition.

    DEPRECATED: This function will be removed in future versions. Ratios will 
    be calculated on derived pupil metrics instead of raw pupil size for 
    interpretable normalization. 

    For each condition's timecourse, calculates the ratio relative to the
    reference condition (default: 'flux') at each matching timepoint. This
    standardizes responses against a reference, enabling comparison across
    conditions. By default, calculates ratio based on baseline-corrected pupil
    size.

    Example: If flux has rel_size_mm=4.0 and melanopsin has 5.0 at time 10s,
    the ratio will be 5.0/4.0 = 1.25.

    Args:
        df: Aggregated pupil data.
        target_col: Column name to calculate ratio for (e.g., 'rel_size_mm', 'pupil_mean').
        condition_col: Column name for condition/trial type.
        grouping_cols: Columns that define unique timecourses to match across conditions.
                      If None, uses [Participant id, Block, Trial time Sec, Trial phase].
        reference_condition: Reference condition to normalize against (default: 'flux').

    Returns:
        DataFrame with additional column:
            - 'norm_{reference_condition}_ratio': Ratio relative to reference condition 
    """
    if target_col not in df.columns:
        raise KeyError(f"Missing target column: {target_col}")
    if condition_col not in df.columns:
        raise KeyError(f"Missing condition column: {condition_col}")

    df = df.copy()

    # Group dataframe by specified columns to identify unique timecourses for matching reference values. Default grouping includes participant, block, time, and phase.
    # Identify unique timecourses that should be matched between conditions
    if grouping_cols is None:
        grouping_cols = []
        if "Participant id" in df.columns:
            grouping_cols.append("Participant id")
        grouping_cols.extend([DEFAULT_BLOCK_COL, DEFAULT_TIME_COL])
        if DEFAULT_PHASE_COL in df.columns:
            grouping_cols.append(DEFAULT_PHASE_COL)

    # Validate grouping columns exist
    for col in grouping_cols:
        if col not in df.columns:
            raise KeyError(f"Missing grouping column: {col}")

    # Extract reference condition timecourse; get only the rows for the reference condition (e.g., 'flux')
    reference_mask = df[condition_col].astype(str).str.lower() == reference_condition.lower()
    reference_timecourse = df[reference_mask][grouping_cols + [target_col]].copy()

    if reference_timecourse.empty:
        raise ValueError(f"No data found for reference condition: '{reference_condition}'")

    # Rename target column to identify it as the reference value
    ref_col_name = f"{target_col}_{reference_condition}_ref"
    reference_timecourse = reference_timecourse.rename(columns={target_col: ref_col_name})

    # Merge reference timecourse to all conditions on the grouping columns to align reference values with each condition's timecourse.
    # I.e. adds reference value as a new column for each row, matched by participant, block, time, and trial phase.
    df = df.merge(reference_timecourse, on=grouping_cols, how="left")

    # Calculate ratio: condition_value / reference_value at each time point and save in new column. E.g., 'flux_norm_ratio' = outcome (e.g. rel_size_mm) / pupil_mean_flux_ref

    # ratio_col = f"{target_col}_{reference_condition}_ratio"
    ratio_col = f"{reference_condition}_norm_ratio"
    df[ratio_col] = df[target_col] / df[ref_col_name]

    # Remove intermediate reference column
    df = df.drop(columns=[ref_col_name])

    return df


def plot_aggregated_traces(
    aggregated: pd.DataFrame,
    time_col: str = DEFAULT_TIME_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    trial_phase_col: str = DEFAULT_PHASE_COL,
    outcome_col: str = "pupil_mean",
    outcome_std_col: str | None = None,
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    output_path: Path | None = None,
) -> None:
    """Plot aggregated traces with optional stim-phase shading.
    
    Args:
        aggregated: Aggregated pupil data.
        time_col: Column name for time.
        condition_col: Column name for condition.
        block_col: Column name for block.
        trial_phase_col: Column name for trial phase.
        outcome_col: Column name for the outcome to plot (e.g., 'pupil_mean', 'rel_size_mm', 'change_size_pct').
        outcome_std_col: Column name for the standard deviation. If None, auto-detects based on outcome_col.
        ylabel: Y-axis label. If None, auto-generates based on outcome_col.
        figsize: Figure size as (width, height). If None, auto-calculated based on panel count.
        output_path: Optional output path for saving the plot.
    """

    # Auto-detect std column if not provided
    if outcome_std_col is None:
        if outcome_col == "pupil_mean" or outcome_col == "rel_size_mm":
            outcome_std_col = "pupil_std"
        else:
            outcome_std_col = None  # No std for normalized columns by default
    
    # Auto-generate ylabel if not provided
    if ylabel is None:
        if outcome_col == "pupil_mean":
            ylabel = "Pupil size (mm)"
        elif outcome_col == "rel_size_mm":
            ylabel = "Rel. pupil size (mm)"
        elif outcome_col == "change_size_pct":
            ylabel = "Pupil size change (%)"
        elif outcome_col.endswith("_ratio"):
            ylabel = "Pupil size (ratio)"
        else:
            ylabel = outcome_col
    
    required_cols = {condition_col, block_col, time_col, outcome_col}
    if outcome_std_col:
        required_cols.add(outcome_std_col)
    missing = required_cols - set(aggregated.columns)
    if missing:
        raise KeyError(f"Aggregated data missing columns: {sorted(missing)}")

    stim_windows = _stim_windows(aggregated, condition_col, block_col, time_col, trial_phase_col)

    conditions = aggregated[condition_col].dropna().unique().tolist()
    blocks = aggregated[block_col].dropna().unique().tolist()
    if not conditions:
        conditions = [None]
    if not blocks:
        blocks = [None]

    n_rows = len(conditions)
    n_cols = len(blocks)

    if figsize is None:
        figsize = (3.5 * n_cols, 2.0 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for idx_cond, condition in enumerate(conditions):
        for idx_block, block in enumerate(blocks):
            ax = axes[idx_cond][idx_block]
            subset = aggregated
            if condition is not None:
                subset = subset[subset[condition_col] == condition]
            if block is not None:
                subset = subset[subset[block_col] == block]

            # Get color for trial type
            color = TRIAL_TYPE_COLORS.get(condition, "tab:blue") if condition else "tab:blue"

            sns.lineplot(
                data=subset,
                x=time_col,
                y=outcome_col,
                ax=ax,
                color=color,
                label=condition.upper() if condition else "Unknown",
            )

            if outcome_std_col and outcome_std_col in subset.columns:
                ax.fill_between(
                    subset[time_col],
                    subset[outcome_col] - subset[outcome_std_col],
                    subset[outcome_col] + subset[outcome_std_col],
                    color=color,
                    alpha=0.2,
                    edgecolor='none',
                )

            window = stim_windows.get((condition, block))
            if window:
                ax.axvspan(window[0], window[1], color="grey", alpha=0.2, zorder=0, linewidth=0)

            # Set title for plot panel
            if idx_cond == 0:
                ax.set_title(f"Block {block}")
            
            ax.set_xlabel(time_col)
            if idx_block == 0:  
                ax.set_ylabel(ylabel)
            ax.legend(loc="best", frameon=True, framealpha=0.9)

    fig.tight_layout(h_pad=0.1, w_pad=0.1)
    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

def plot_condition_means_across_blocks(
    aggregated: pd.DataFrame,
    time_col: str = DEFAULT_TIME_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    trial_phase_col: str = DEFAULT_PHASE_COL,
    participant_col: str = "Participant id",
    outcome_col: str = "pupil_mean",
    ylabel: str | None = None,
    figsize: tuple[float, float] | None = None,
    output_path: Path | None = None,
    individual_lines: bool = False,
) -> None:
    """Plot condition means averaged across blocks with separate panels per condition.
    
    Averages data across all blocks for each condition, then plots each condition
    in a separate subplot panel. Shows mean ± std error across participants.
    
    Args:
        aggregated: Aggregated pupil data.
        time_col: Column name for time.
        condition_col: Column name for condition.
        block_col: Column name for block.
        trial_phase_col: Column name for trial phase.
        participant_col: Column name for participant ID.
        outcome_col: Column name for the outcome to plot (e.g., 'pupil_mean', 'rel_size_mm', 'change_size_pct').
        ylabel: Y-axis label. If None, auto-generates based on outcome_col.
        figsize: Figure size as (width, height). If None, auto-calculated based on panel count.
        output_path: Optional output path for saving the plot.
        individual_lines: Whether to plot individual lines for each participant.
    """
    # Auto-generate ylabel if not provided
    if ylabel is None:
        if outcome_col == "pupil_mean":
            ylabel = "Pupil size (mm)"
        elif outcome_col == "rel_size_mm":
            ylabel = "Rel. pupil size (mm)"
        elif outcome_col == "change_size_pct":
            ylabel = "Pupil size change (%)"
        elif outcome_col.endswith("_ratio"):
            ylabel = "Pupil size (ratio)"
        else:
            ylabel = outcome_col

    required_cols = {condition_col, time_col, outcome_col}
    if participant_col in aggregated.columns:
        required_cols.add(participant_col)
    missing = required_cols - set(aggregated.columns)
    if missing:
        raise KeyError(f"Aggregated data missing columns: {sorted(missing)}")

    # Average across blocks for each condition, participant, and timepoint
    # and then over block-averages --> this way we give equal weight to each
    # block (instead of weighting by number of trials per block)
    group_cols = [condition_col, time_col]
    if participant_col in aggregated.columns:
        group_cols.insert(1, participant_col)
    if trial_phase_col in aggregated.columns:
        group_cols.append(trial_phase_col)

    block_averaged = (
        aggregated.groupby(group_cols, dropna=False)[outcome_col]
        .mean()
        .reset_index()
        .rename(columns={outcome_col: f"{outcome_col}_block_avg"})
    )

    # Get average stim window across blocks for each condition
    stim_windows_all = _stim_windows(aggregated, condition_col, block_col, time_col, trial_phase_col)
    avg_stim_windows = {}
    for (condition, block), (start, end) in stim_windows_all.items():
        if condition not in avg_stim_windows:
            avg_stim_windows[condition] = []
        avg_stim_windows[condition].append((start, end))

    # Average start and end times across blocks
    for condition in avg_stim_windows:
        windows = avg_stim_windows[condition]
        avg_start = np.mean([w[0] for w in windows])
        avg_end = np.mean([w[1] for w in windows])
        avg_stim_windows[condition] = (avg_start, avg_end)

    conditions = sorted(block_averaged[condition_col].dropna().unique())
    n_conditions = len(conditions)
    n_total = n_conditions + 1  # Add one for global mean

    # Determine layout (prefer horizontal layout for small number of conditions)
    if n_total <= 3:
        n_rows, n_cols = 1, n_total
    elif n_total == 4:
        n_rows, n_cols = 2, 2
    else:
        n_cols = 3
        n_rows = int(np.ceil(n_total / n_cols))

    if figsize is None:
        figsize = (4.0 * n_cols, 3.0 * n_rows)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # Plot each condition
    for idx, condition in enumerate(conditions):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        # Filter to this condition and compute mean ± SEM across participants
        condition_block_avg = block_averaged[block_averaged[condition_col] == condition]

        # Compute population mean and SEM for this condition
        group_cols_for_stats = [time_col]
        if trial_phase_col in condition_block_avg.columns:
            group_cols_for_stats.append(trial_phase_col)

        condition_stats = (
            condition_block_avg.groupby(group_cols_for_stats, dropna=False)[f"{outcome_col}_block_avg"]
            .agg(mean="mean", sem="sem")
            .reset_index()
        )

        # Get color for trial type
        color = TRIAL_TYPE_COLORS.get(condition, "tab:blue")

        # Plot individual participant traces (block-averaged) as transparent lines
        if individual_lines and participant_col in condition_block_avg.columns:
            participants = condition_block_avg[participant_col].unique()

            for participant in participants:
                participant_data = condition_block_avg[condition_block_avg[participant_col] == participant]
                ax.plot(
                    participant_data[time_col],
                    participant_data[f"{outcome_col}_block_avg"],
                    color=color,
                    alpha=0.5,
                    linewidth=0.5,
                    zorder=1,
                )

        # Plot mean line
        ax.plot(
            condition_stats[time_col],
            condition_stats["mean"],
            color=color,
            linewidth=2,
            label=condition.upper(),
            zorder=3,
        )

        # Add shaded error region (mean ± SEM)
        ax.fill_between(
            condition_stats[time_col],
            condition_stats["mean"] - condition_stats["sem"],
            condition_stats["mean"] + condition_stats["sem"],
            color=color,
            alpha=0.3,
            zorder=2,
            edgecolor='none',
        )

        # Add stim window shading
        if condition in avg_stim_windows:
            window = avg_stim_windows[condition]
            ax.axvspan(window[0], window[1], color="grey", alpha=0.2, zorder=0, linewidth=0)

        ax.set_title(condition.upper())
        ax.set_xlabel("Trial time [s]")
        if col == 0:
            ax.set_ylabel(ylabel)
        ax.legend(loc="lower right", frameon=False, framealpha=0.9)

    # Plot global mean in the last subplot
    global_idx = n_total - 1
    row = global_idx // n_cols
    col = global_idx % n_cols
    ax = axes[row][col]

    # Compute global mean and SEM across all conditions (already block-averaged)
    group_cols_for_stats = [time_col]
    if trial_phase_col in block_averaged.columns:
        group_cols_for_stats.append(trial_phase_col)

    global_stats = (
        block_averaged.groupby(group_cols_for_stats, dropna=False)[f"{outcome_col}_block_avg"]
        .agg(mean="mean", sem="sem")
        .reset_index()
    )

    # Plot global mean line ---
    ax.plot(
        global_stats[time_col],
        global_stats["mean"],
        color="black",
        linewidth=2,
        label="Global mean",
        zorder=3,
    )
    # Add shaded error region (mean ± SEM)
    ax.fill_between(
        global_stats[time_col],
        global_stats["mean"] - global_stats["sem"],
        global_stats["mean"] + global_stats["sem"],
        color="black",
        alpha=0.3,
        zorder=2,
        edgecolor='none',
    )
    # Add stim window shading for global mean (average across all conditions)
    if avg_stim_windows:
        starts = [w[0] for w in avg_stim_windows.values()]
        ends = [w[1] for w in avg_stim_windows.values()]
        global_start = np.mean(starts)
        global_end = np.mean(ends)
        ax.axvspan(global_start, global_end, color="grey", alpha=0.2, zorder=0, linewidth=0)
    ax.set_title("Global mean")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", frameon=False, framealpha=0.9)

    # Hide unused subplots
    for idx in range(n_total, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle("Condition-specific pupil trace averaged across blocks and participants", fontsize=14, y=0.995)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300)
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()

def aggregate_pupil_traces(
    csv_path: Path,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    time_col: str = DEFAULT_TIME_COL,
    output_csv: Path | None = None,
    figsize: tuple[float, float] | None = None,
    plot_output_path: Path | None = None,
    normalize_baseline: bool = True,
    baseline_method: str = "mean",
) -> pd.DataFrame:
    """Process participant data with aggregation and optional plotting.

    Loads the participant CSV, aggregates traces by condition/block/time/phase, optionally
    normalizes to a pre-stimulus baseline, writes the aggregated table, and can generate
    a plot of aggregated traces. Function will create new columns `pupil_mean` and `pupil_std` in the output DataFrame; and if normalisation is applied, also `baseline_mm`, `rel_size_mm`, and `change_size_pct`.

    This function can be called either from the command line (via main()) or
    imported and used programmatically.

    Args:
        csv_path: Path to participant final CSV.
        condition_col: Column name for condition.
        block_col: Column name for block.
        time_col: Column name for trial time (seconds).
        output_csv: Optional path to write aggregated CSV.
        figsize: Figure size as (width, height). If None, auto-calculated.
        plot_output_path: Optional output path for the plot (e.g., plot.png).
        normalize_baseline: Whether to normalize to pre-stim baseline.
        baseline_method: Method to calculate baseline ('mean' or 'median').

    Returns:
        Aggregated DataFrame.
    """
    df = load_participant_csv(csv_path)
    grouping_cols = [condition_col, block_col, time_col, DEFAULT_PHASE_COL]
    result = aggregate_participant_traces(df, grouping_cols=grouping_cols)

    if normalize_baseline:
        result = normalise_to_baseline(
            result,
            trial_phase_col=DEFAULT_PHASE_COL,
            condition_col=condition_col,
            block_col=block_col,
            baseline_method=baseline_method,
        )
    # Note: Ratio to reference condition is done on pupil metrics, not on raw pupil size.
    #
    # Calculate ratio to reference condition (flux)
    # result = depr_generate_ratio_to_reference(
    #     result,
    #     condition_col=condition_col,
    #     grouping_cols=["Participant id", block_col, time_col, DEFAULT_PHASE_COL] if "Participant id" in result.columns else [block_col, time_col, DEFAULT_PHASE_COL],
    #     reference_condition="flux",
    # )
    #
    if output_csv:
        result.to_csv(output_csv, index=False)

    if plot_output_path:
        # outcome_col: Column name for the outcome to plot (e.g., 'pupil_mean', 'rel_size_mm', 'change_size_pct'
        plot_aggregated_traces(
            result,
            time_col=time_col,
            outcome_col="rel_size_mm",
            figsize=figsize,
            output_path=plot_output_path,
        )

    return result


def plot_participant_traces_conditionwise(
    data: pd.DataFrame,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
    time_col: str = DEFAULT_TIME_COL,
    trial_phase_col: str = DEFAULT_PHASE_COL,
    participant_col: str = "Participant id",
    outcome_col: str = "pupil_mean",
    outcome_std_col: str | None = None,
    ylabel: str | None = None,
    figsize_per_condition: tuple[float, float] | None = None,
    output_dir: Path | None = None,
) -> None:
    """Plot all participants with one figure per trial type.
    
    Each figure has subplots in participant x block layout.
    
    Args:
        aggregated: Aggregated pupil data for all participants.
        condition_col: Column name for condition/trial type.
        block_col: Column name for block.
        time_col: Column name for time.
        trial_phase_col: Column name for trial phase.
        participant_col: Column name for participant ID.
        outcome_col: Column name for the outcome to plot (e.g., 'pupil_mean', 'rel_size_mm', 'change_size_pct').
        outcome_std_col: Column name for the standard deviation. If None, auto-detects based on outcome_col.
        ylabel: Y-axis label. If None, auto-generates based on outcome_col.
        figsize_per_condition: Figure size per condition. If None, auto-calculated.
        output_dir: Optional directory to save plots (one per condition).
    """
    # Auto-detect std column if not provided
    if outcome_std_col is None:
        if outcome_col == "pupil_mean":
            outcome_std_col = "pupil_std"
        else:
            outcome_std_col = None  # No std for normalized columns by default

    # Auto-generate ylabel if not provided
    if ylabel is None:
        if outcome_col == "pupil_mean":
            ylabel = "Pupil size (mm)"
        elif outcome_col == "rel_size_mm":
            ylabel = "Rel. pupil size (mm)"
        elif outcome_col == "change_size_pct":
            ylabel = "Pupil size change (%)"
        else:
            ylabel = outcome_col

    if condition_col not in data.columns:
        raise KeyError(f"Missing condition column: {condition_col}")
    if participant_col not in data.columns:
        raise KeyError(f"Missing participant column: {participant_col}")

    # Calculate stim windows from aggregated data
    stim_windows = _stim_windows(data, condition_col, block_col, time_col, trial_phase_col)

    conditions = sorted(data[condition_col].dropna().unique())
    participants = sorted(data[participant_col].dropna().unique())
    blocks = sorted(data[block_col].dropna().unique())

    for _, cond in enumerate(conditions):
        cond_df = data[data[condition_col] == cond]

        n_rows = len(participants)
        n_cols = len(blocks)

        if figsize_per_condition is None:
            figsize = (2.75 * n_cols, 1.75 * n_rows)
        else:
            figsize = figsize_per_condition

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=figsize,
            sharex=True,
            sharey=True,
            squeeze=False,
        )

        # Get color for trial type or default to blue
        color = TRIAL_TYPE_COLORS.get(cond, "tab:blue")

        for idx_pid, participant in enumerate(participants):
            for idx_block, block in enumerate(blocks):
                ax = axes[idx_pid][idx_block]
                subset = cond_df[
                    (cond_df[participant_col] == participant) &
                    (cond_df[block_col] == block)
                ]

                if not subset.empty:
                    sns.lineplot(
                        data=subset,
                        x=time_col,
                        y=outcome_col,
                        ax=ax,
                        color=color,
                    )

                    if outcome_std_col and outcome_std_col in subset.columns:
                        ax.fill_between(
                            subset[time_col],
                            subset[outcome_col] - subset[outcome_std_col],
                            subset[outcome_col] + subset[outcome_std_col],
                            color=color,
                            alpha=0.2,
                            edgecolor='none',
                        )

                    window = stim_windows.get((cond, block))
                    if window:
                        ax.axvspan(window[0], window[1], color="grey", alpha=0.2, zorder=0, linewidth=0)

                # Label axes
                if idx_block == 0:
                    ax.set_ylabel(f"P{participant}\n{ylabel}")
                else:
                    ax.set_ylabel("")

                if idx_pid == 0:
                    ax.set_title(f"Block {block}")

                if idx_pid == n_rows - 1:
                    ax.set_xlabel(time_col)
                else:
                    ax.set_xlabel("")

        fig.suptitle(f"Trial type: {cond}", fontsize=14, y=0.995)
        fig.tight_layout(h_pad=0.1, w_pad=0.1)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{cond}_all_participants.png"
            fig.savefig(output_path, dpi=300)
            print(f"Saved plot to: {output_path}")
            plt.close(fig)
        else:
            plt.show()


def process_all_participants_data(
    data_dir: Path | None = None,
    pattern: str = "[0-9][0-9][0-9]_*.csv",
    output_csv: Path | None = DEFAULT_PROJECT_DIR / "data" / "aggregated" / "data.csv",
    plot_output_dir: Path | None = None,
    normalize_baseline: bool = True,
    baseline_method: str = "mean",
    trial_count_tex: Path | None = DEFAULT_TRIAL_COUNT_TEX,
) -> pd.DataFrame:
    """Process all participants' data with aggregation and optional plotting.

    Memory-efficient: processes each participant individually using process_participant_data(),
    then combines results. Each participant's data is aggregated, baseline-normalized, and
    ratio-calculated before combining.

    Args:
        data_dir: Directory containing participant CSVs.
        pattern: Glob pattern to find participant files.
        output_csv: Optional path to write aggregated CSV.
        plot_output_dir: Optional directory to save plots (one per condition).
        normalize_baseline: Whether to normalize to pre-stim baseline, default is True.
        baseline_method: Method to calculate baseline ('mean' or 'median').
        trial_count_tex: Optional path to write accepted-trial-count LaTeX macros.

    Returns:
        Aggregated DataFrame for all participants.
    """

    data_dir = data_dir or DEFAULT_FINAL_DIR
    csv_paths = list(_iter_csvs(data_dir, pattern))

    if not csv_paths:
        raise FileNotFoundError(f"No participant CSVs found in: {data_dir} (pattern: {pattern})")

    # Process each participant individually (aggregation + baseline normalization + ratio calculation)
    processed_results = []

    for csv_path in csv_paths:
        participant_df = aggregate_pupil_traces(
            csv_path=csv_path,
            output_csv=None,  # Don't save individual participant CSVs
            normalize_baseline=normalize_baseline,
            baseline_method=baseline_method,
            plot_output_path=None,  # Don't save individual participant plots
        )
        processed_results.append(participant_df)

    # Combine all processed results
    result = pd.concat(processed_results, ignore_index=True)

    # Validate that expected columns are present in the combined data
    required_cols = {DEFAULT_CONDITION_COL, DEFAULT_BLOCK_COL, DEFAULT_TIME_COL}
    missing = required_cols - set(result.columns)
    if missing:
        raise KeyError(f"Combined data missing expected columns: {sorted(missing)}")

    print("\nFiltering participants with insufficient block coverage...")
    result = filter_min_blocks(result)

    # # Print summary of N (participants) and N (Observations) per condition
    # for condition in result[DEFAULT_CONDITION_COL].unique():
    #     cond_data = result[result[DEFAULT_CONDITION_COL] == condition]
    #     n_participants = cond_data["Participant id"].nunique() if "Participant id" in cond_data.columns else "N/A"
    #     # Unique participant x block combinations as observations
    #     n_observations = cond_data.drop_duplicates(subset=["Participant id", "Block"]).shape[0]
    #     print(f"Condition: {condition}")
    #     print(f"\tNumber of participants: {n_participants}")
    #     print(f"\tNumber of observations: {n_observations}")

    if output_csv:
        result.to_csv(output_csv, index=False)
        print(f"Saved CSV to: {output_csv}")

        # Save metadata
        metadata_path = output_csv.parent / f"{output_csv.stem}_metadata.csv"
        save_aggregated_data_metadata(
            output_path=metadata_path,
            include_normalized=normalize_baseline,
            include_ratio=True  # Ratio is always calculated in process_participant_data
        )

    if trial_count_tex:
        save_trial_count_variables(result, csv_paths, trial_count_tex)

    if plot_output_dir:
        plot_condition_means_across_blocks(
            result,
            outcome_col="rel_size_mm",
            output_path=plot_output_dir / "fig_global_condition_means.png"
        )

    return result


def save_trial_count_variables(
    aggregated: pd.DataFrame,
    csv_paths: list[Path],
    output_path: Path,
) -> pd.DataFrame:
    r"""Count accepted trials in the analysed traces and write them as LaTeX macros.

    Trials are counted as distinct trial numbers in the final per-participant
    CSVs, restricted to the participant x condition x block traces retained in
    `aggregated`. All runs recorded in a block are pooled, so a block can hold
    more trials than the SCHEDULED_RUNS_PER_BLOCK runs contain (a run repeated
    after signal loss keeps the valid trials recorded before the interruption).

    Writes ``\AggregatedTrials<Statistic>`` macros, e.g.
    ``\AggregatedTrialsPerTraceMedian``.

    Returns:
        DataFrame with one row per retained participant x condition x block and
        its accepted-trial count.
    """
    keys = ["Participant id", DEFAULT_CONDITION_COL, DEFAULT_BLOCK_COL]
    trials = pd.concat(
        [pd.read_csv(path, usecols=keys + [DEFAULT_TRIAL_NO_COL]) for path in csv_paths],
        ignore_index=True,
    ).drop_duplicates()

    traces = aggregated[keys].drop_duplicates()
    blocks = traces[["Participant id", DEFAULT_BLOCK_COL]].drop_duplicates()

    per_trace = (
        trials.groupby(keys)[DEFAULT_TRIAL_NO_COL].nunique().rename("n_trials").reset_index()
        .merge(traces, on=keys, how="inner")
    )
    if len(per_trace) != len(traces):
        raise ValueError("Some retained traces have no trials in the final CSVs.")

    # All conditions of a block share its runs, so count every trial of the block.
    per_block = (
        trials.groupby(["Participant id", DEFAULT_BLOCK_COL])[DEFAULT_TRIAL_NO_COL].nunique()
        .rename("n_trials").reset_index()
        .merge(blocks, on=["Participant id", DEFAULT_BLOCK_COL], how="inner")
    )
    scheduled = SCHEDULED_RUNS_PER_BLOCK * TRIALS_PER_RUN

    median = per_trace["n_trials"].median()
    stats = pd.DataFrame([{
        "statistic": "trials",
        "per_trace_min": int(per_trace["n_trials"].min()),
        "per_trace_median": f"{median:g}",
        "per_trace_max": int(per_trace["n_trials"].max()),
        "n_traces": len(per_trace),
        "n_blocks": len(per_block),
        "n_blocks_above_scheduled": int((per_block["n_trials"] > scheduled).sum()),
    }])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        format_latex_variables(stats, row_key_col="statistic", prefix="Aggregated"),
        encoding="utf-8",
    )
    print(f"Saved LaTeX variables: {output_path}")
    return per_trace


def save_aggregated_data_metadata(
    output_path: Path,
    include_normalized: bool = True,
    include_ratio: bool = True
) -> None:
    """Create and save metadata describing aggregated data columns.
    
    Args:
        output_path: Path where metadata CSV will be saved.
        include_normalized: Whether normalized columns are present in the data.
        include_ratio: Whether flux ratio column is present in the data.
    """
    metadata = [
        {
            "column_name": "Participant id",
            "description": "Unique participant identifier",
            "unit": "categorical"
        },
        {
            "column_name": "Trial type",
            "description": "Experimental condition/stimulus type (e.g., melanopsin, s, l-m, lms, flux)",
            "unit": "categorical"
        },
        {
            "column_name": "Block",
            "description": "Experimental block number",
            "unit": "integer"
        },
        {
            "column_name": "Trial time Sec",
            "description": "Time within trial in seconds",
            "unit": "seconds"
        },
        {
            "column_name": "Trial phase",
            "description": "Phase of trial (pre-stim, stim, post-stim)",
            "unit": "categorical"
        },
        {
            "column_name": "pupil_mean",
            "description": "Mean pupil diameter (absolute size, averaged across all accepted trials of the block)",
            "unit": "mm"
        },
        {
            "column_name": "pupil_std",
            "description": "Standard deviation of pupil diameter across accepted trials",
            "unit": "mm"
        },
        {
            "column_name": "n",
            "description": "Number of accepted trials with data at this time point",
            "unit": "count"
        }
    ]
    
    if include_normalized:
        metadata.extend([
            {
                "column_name": "baseline_mm",
                "description": "Baseline pupil diameter calculated from pre-stim phase",
                "unit": "mm"
            },
            {
                "column_name": "rel_size_mm",
                "description": "Baseline-corrected pupil diameter (pupil_mean - baseline_mm)",
                "unit": "mm"
            },
            {
                "column_name": "change_size_pct",
                "description": "Percentage change from baseline ((pupil_mean - baseline_mm) / baseline_mm * 100)",
                "unit": "%"
            }
        ])
    
    if include_ratio:
        metadata.append({
            "column_name": "flux_norm_ratio",
            "description": "Flux-normalized ratio (pupil_mean / flux_pupil_mean at matching timepoint)",
            "unit": "ratio"
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_path, index=False)
    print(f"Saved metadata to: {output_path}")


def main_all(plot_by_condition: bool = True, plot_by_participant: bool = True) -> pd.DataFrame:
    """Process all participants and generate aggregate plots.

    Returns:
        pd.DataFrame: Aggregated DataFrame for all participants.
    """

    print("Processing all participants and generating aggregate plots...")

    # Ensure default plot directory exists
    DEFAULT_FIG_DIR.mkdir(parents=True, exist_ok=True)

    csv_paths = list(_iter_csvs(DEFAULT_FINAL_DIR, "*.csv"))

    for csv_path in csv_paths:
        # create aggregated DataFrame for this participant with normalisation, plotting is done below
        df = aggregate_pupil_traces(
            csv_path=csv_path,
            normalize_baseline=True,
            baseline_method="mean",
            plot_output_path=None,
        )

        if plot_by_participant:
            # mean pupil
            output_path = DEFAULT_FIG_DIR / "aggtraces_by_participant" / "pupil_mean" / f"aggtraces_pupil_{csv_path.stem.split('_', 1)[0]}.png"
            output_path.parent.mkdir(parents=False, exist_ok=True)
            plot_aggregated_traces(
                aggregated=df,
                outcome_col="pupil_mean",
                output_path=output_path,
            )
            # normalised traces (baseline-corrected in mm)
            output_path = (DEFAULT_FIG_DIR / "aggtraces_by_participant" / "norm" / f"aggtraces_pupil_norm_{csv_path.stem.split('_', 1)[0]}.png")
            output_path.parent.mkdir(parents=False, exist_ok=True)
            plot_aggregated_traces(
                aggregated=df,
                outcome_col="rel_size_mm",
                output_path=output_path,
            )
            # normalised traces in percentage change from baseline
            output_path = (DEFAULT_FIG_DIR / "aggtraces_by_participant" / "pct" / f"aggtraces_pupil_pct_{csv_path.stem.split('_', 1)[0]}.png")
            output_path.parent.mkdir(parents=False, exist_ok=True)  
            plot_aggregated_traces(
                aggregated=df,
                outcome_col="change_size_pct",
                output_path=output_path,
            )
            # ratio traces in percentage change
            # output_path = DEFAULT_FIG_DIR / "aggtraces_by_participant_ratio" / f"aggtraces_pupil_ratio_{csv_path.stem.split('_', 1)[0]}.png"
            # output_path.parent.mkdir(parents=True, exist_ok=True)
            # plot_aggregated_traces(
            #     aggregated=df,
            #     outcome_col="flux_norm_ratio",
            #     output_path=output_path,
            # )

    # ALL PARTICIPANTS
    output_path = DEFAULT_PROJECT_DIR / "data" / "aggregated" / "data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggdf = process_all_participants_data(
        data_dir=DEFAULT_FINAL_DIR,
        normalize_baseline=True,
        baseline_method="mean",
        output_csv=output_path,
        plot_output_dir=DEFAULT_FIG_DIR / "aggtraces_by_condition",
    )

    if plot_by_condition:
        # pupil trace
        plot_participant_traces_conditionwise(
            data=aggdf,
            outcome_col="pupil_mean",
            output_dir=DEFAULT_FIG_DIR / "aggtraces_by_condition" / "pupil_mean",
        )
        # normalised plot
        plot_participant_traces_conditionwise(
            data=aggdf,
            outcome_col="rel_size_mm",
            output_dir=DEFAULT_FIG_DIR / "aggtraces_by_condition" / "norm",
        )
        # normalised traces in percentage change
        plot_participant_traces_conditionwise(
            data=aggdf,
            outcome_col="change_size_pct",
            output_dir=DEFAULT_FIG_DIR / "aggtraces_by_condition" / "pct",
        )

        # ratio traces in normalised pupil diameter
        plot_condition_means_across_blocks(
            aggregated=aggdf,
            outcome_col="change_size_pct",
            output_path=DEFAULT_FIG_DIR / "aggtraces_by_condition" / "means_across_blocks.png",
        )

    # Print summary of the aggregated DataFrame
    aggdf.info()

    return aggdf


if __name__ == "__main__":
    """Command-line interface for aggregating pupil traces."""
    parser = argparse.ArgumentParser(
        description="Aggregate a participant's pupil traces by condition and block."
    )
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_FINAL_DIR, help="Path to participant CSV or directory containing participant CSVs (default: process all in data/final).")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write aggregated CSV.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional output path for the plot (e.g., plot.png).",
    )
    args = parser.parse_args()

    if args.csv_path.is_dir():
        # If a directory is provided, process all participants
        main_all(
            plot_by_condition=True,
            plot_by_participant=False,
        )
    else:
        # Process a single participant
        result = aggregate_pupil_traces(
            csv_path=args.csv_path,
            output_csv=args.output,
            plot_output_path=args.plot_output,
        )

        if not args.output:
            print(result)
