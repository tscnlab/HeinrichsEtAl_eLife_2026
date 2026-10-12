"""Load a participant CSV into a single DataFrame."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_FINAL_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "retinawise_results" / "final"
)

def _iter_csvs(data_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(data_dir.glob(pattern))


def _participant_id_from_path(path: Path) -> int:
    return int(path.name.split("_", maxsplit=1)[0])


def load_participant_csv(csv_path: Path) -> pd.DataFrame:
    """Load a participant CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    if "participant_id" not in df.columns:
        df.insert(0, "participant_id", _participant_id_from_path(csv_path))
    return df


def load_all_participant_data(
    data_dir: Path | None = None,
    pattern: str = "[0-9][0-9][0-9]_*.csv",
) -> pd.DataFrame:
    """Load all participants' CSVs into one DataFrame.

    Args:
        data_dir: Directory containing participant CSVs.
        pattern: Glob pattern used to find participant CSVs.

    Returns:
        Combined DataFrame with a `Participant id` column.
    """

    data_dir = data_dir or DEFAULT_FINAL_DIR
    csv_paths = list(_iter_csvs(data_dir, pattern))
    if not csv_paths:
        raise FileNotFoundError(
            f"No participant CSVs found in: {data_dir} (pattern: {pattern})"
        )

    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        df = load_participant_csv(csv_path=csv_path)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a single participant's data into one DataFrame."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_FINAL_DIR,
        help="Directory containing participant CSV files.",
    )
    parser.add_argument(
        "participant_id",
        type=int,
        nargs="?",
        default=201,
        help="Participant identifier used in the file name (default: 201).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Glob pattern used to find participant CSVs (defaults to '<id>_*.csv' with 3-digit id).",
    )
    args = parser.parse_args()

    # iterate over all CSVs in the directory matching the pattern, but only load the one for the specified participant_id
    pattern = args.pattern or rf"{args.participant_id:03d}_*.csv"
    df = load_participant_csv( 
        csv_path=args.data_dir / pattern
    )
    print(df)
    print(f"\nLoaded {df.shape[0]} rows from participant [{args.participant_id}].")


if __name__ == "__main__":
    main()
