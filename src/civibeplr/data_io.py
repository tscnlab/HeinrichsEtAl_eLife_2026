"""Data loading utilities for pupillometry analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_AGGREGATED_DATA = DEFAULT_PROJECT_DIR / "data" / "aggregated" / "data.csv"

# Column name constants
DEFAULT_PARTICIPANT_COL = "Participant id"
DEFAULT_CONDITION_COL = "Trial type"
DEFAULT_BLOCK_COL = "Block"
DEFAULT_TIME_COL = "Trial time Sec"
DEFAULT_TRIALPHASE_COL = "Trial phase"


def filter_min_blocks(
    df: pd.DataFrame,
    min_blocks: int = 6,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
    block_col: str = DEFAULT_BLOCK_COL,
) -> pd.DataFrame:
    """Remove participant-condition pairs with fewer than *min_blocks* distinct blocks.

    For each condition separately, the number of distinct blocks per participant
    is counted.  Participants below the threshold in a given condition have all
    their rows for that condition dropped.
    """
    block_counts = (
        df.groupby([condition_col, participant_col])[block_col]
        .nunique()
        .reset_index()
        .rename(columns={block_col: "n_blocks"})
    )
    sufficient = block_counts[block_counts["n_blocks"] >= min_blocks][
        [condition_col, participant_col]
    ]

    all_combos  = set(zip(block_counts[condition_col], block_counts[participant_col]))
    kept_combos = set(zip(sufficient[condition_col],   sufficient[participant_col]))
    excluded    = sorted(all_combos - kept_combos)
    if excluded:
        print(f"  Excluding {len(excluded)} participant-condition pairs "
              f"with < {min_blocks} blocks:")
        for cond, pid in excluded:
            n = int(block_counts.loc[
                (block_counts[condition_col] == cond) &
                (block_counts[participant_col] == pid), "n_blocks"
            ].iloc[0])
            print(f"    participant {pid}, condition '{cond}': {n} block(s)")
    else:
        print(f"  All participants have >= {min_blocks} blocks in every condition.")

    return df.merge(sufficient, on=[condition_col, participant_col], how="inner")


def load_aggregated_data(data_path: Path | None = None) -> pd.DataFrame:
    """Load aggregated pupil data.
    
    Args:
        data_path: Path to aggregated data CSV. If None, uses default location.
        
    Returns:
        DataFrame with aggregated pupil data.
    """
    data_path = data_path or DEFAULT_AGGREGATED_DATA
    
    if not data_path.exists():
        raise FileNotFoundError(f"Aggregated data not found at: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"Loaded aggregated data: {len(df)} rows, {len(df.columns)} columns")
    print(f"Participants: {df[DEFAULT_PARTICIPANT_COL].nunique() if DEFAULT_PARTICIPANT_COL in df.columns else 'Unknown'}")
    print(f"Conditions: {df[DEFAULT_CONDITION_COL].unique().tolist() if DEFAULT_CONDITION_COL in df.columns else 'Unknown'}")
    
    return df
