"""Centralized constants for analysis scripts."""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_DIR = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _PROJECT_DIR / "results" / "tables"

DEFAULT_PUPIL_PATH  = _RESULTS_DIR / "pupil_cosinor" / "pupil_cosinor_2step_param.csv"
DEFAULT_SALIVA_PATH = _RESULTS_DIR / "saliva_cosinor" / "saliva_cosinor_best_fit.csv"

# ── Stimulus window constants ─────────────────────────────────────────────────

DEFAULT_T_STIM_START    = 0.0   # seconds
DEFAULT_T_STIM_END      = 5.0   # seconds
DEFAULT_ADD_POSTSTIM_SEC = 0  # extra post-stimulus seconds included in the cosinor fit window

# ── Column names ──────────────────────────────────────────────────────────────

# Raw pupil CSV columns
PUPIL_PARTICIPANT_COL = "Participant id"
PUPIL_CONDITION_COL   = "Trial type"
PUPIL_BLOCK_COL       = "Block"

# Raw saliva CSV column
SALIVA_PARTICIPANT_COL = "participant_id"

# After merge / rename, participant column used in models (no spaces → safe in formulas)
ANALYSIS_PARTICIPANT_COL = "participant_id"

# Saliva columns to bring into the merged table
SALIVA_COLS_KEEP = [
    SALIVA_PARTICIPANT_COL,
    "tau_hours",       # individual best-fit circadian period [h]
    "rsquared",        # saliva model R²
]

# ── Protocol constants ────────────────────────────────────────────────────────

BLOCK_INTERVAL_HOURS = 3.75   # hours between consecutive block start times

if __name__ == "__main__":
    print("This module defines constants for analysis scripts. It is not meant to be run directly.")

# ── Human-readable labels for parameter and time columns ─────────────────────

PARAMETER_TITLE: dict[str, str] = {
    # Pupil cosinor outcome variables
    "log_amplitude": "Log amplitude (fundamental)",
    "log_amplitude_2f": "Log amplitude (2nd harmonic)",
    "amplitude": "Amplitude (fundamental)",
    "amplitude_2f": "Amplitude (2nd harmonic)",
    "phase": "Phase (fundamental)",
    "phase_2f": "Phase (2nd harmonic)",
    "phase_rad": "Phase (fundamental)",
    "phase_rad_2f": "Phase (2nd harmonic)",
    "phase_hours": "Estimated peak time (fundamental) in hours",
    "phase_hours_2f": "Estimated peak time (2nd harmonic) in hours",
    "tonic_min": "Tonic minimum",
    "tonic_mean": "Tonic mean",
    # Protocol-specified FLUX-normalised outcome (supplementary branch).
    # Additive keys: they cannot collide with any primary outcome name.
    "plr_ratio": "FLUX-normalised maximum constriction (R)",
    "log_plr_ratio": "Log FLUX-normalised maximum constriction (log R)",
}

PARAMETER_LABELS: dict[str, str] = {
    # Pupil cosinor outcome variables
    "log_amplitude": "Log amplitude\n(fundamental) [mm]",
    "log_amplitude_2f": "Log amplitude\n(2nd harmonic) [mm]",
    "amplitude": "Amplitude\n(fundamental) [mm]",
    "amplitude_2f": "Amplitude\n(2nd harmonic) [mm]",
    "phase": "Phase\n(fundamental) [rad]",
    "phase_2f": "Phase\n(2nd harmonic) [rad]",
    "phase_rad": "Phase\n(fundamental) [rad]",
    "phase_rad_2f": "Phase\n(2nd harmonic) [rad]",
    "phase_hours": "Estimated peak time\n(fundamental) [h]",
    "phase_hours_2f": "Estimated peak time\n(2nd harmonic) [h]",
    "tonic_min": "Tonic minimum [mm]",
    "tonic_mean": "Tonic mean [mm]",
    # Protocol-specified FLUX-normalised outcome (supplementary branch).
    "plr_ratio": "R = condition / FLUX\n[dimensionless]",
    "log_plr_ratio": "log R = log(condition / FLUX)\n[dimensionless]",
    # Saliva cosinor parameters
    "tau_hours": "Circadian period τ [h]",
    "rsquared": "R²",
    "adjrsquared": "Adjusted R²",
    "aic": "AIC",
    "bic": "BIC",
    "rmse": "RMSE",
    # Time / block variables
    "hours_since_start": "Hours since start [h]",
    "Block": "Block",
}


def lowercase_first_word(text: str) -> str:
    """Lowercase the first word for use mid-sentence, unless it is an acronym.

    "Tonic minimum" -> "tonic minimum", but "FLUX-normalised ..." is unchanged.
    """
    if len(text) > 1 and text[:2].isupper():
        return text
    return text[:1].lower() + text[1:]


def get_parameter_label(col: str, title=False) -> str:
    """Return a human-readable axis label for a parameter column name."""
    if title:
        out = PARAMETER_TITLE.get(col, col)
    else:
        out = PARAMETER_LABELS.get(col, col)
    return out


# ── Valid outcome column names ────────────────────────────────────────────────

PARAMETER_CHOICES = [
    "baseline",     # baseline pupil size; pre-stimulus mean
    "amplitude",  # fundamental amplitude
    "amplitude_2f",  # 2nd harmonic amplitude
    "phase",        # 2nd harmonic amplitude
    "phase_2f",     # 2nd harmonic amplitude
    "tonic_min",     # tonic minimum during stimulus
    #"tonic_mean",  # tonic mean during stimulus
    "log_amplitude",  # log-transformed fundamental amplitude
]


# ═══════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════

def load_pupil_results(path: Path | None = None) -> pd.DataFrame:
    """Load the two-step cosinor results CSV."""
    if path is None:
        path = DEFAULT_PUPIL_PATH
    return pd.read_csv(path)


def load_saliva_results(path: Path | None = None) -> pd.DataFrame:
    """Load the saliva cosinor best-fit estimates CSV."""
    if path is None:
        path = DEFAULT_SALIVA_PATH
    return pd.read_csv(path)


# ═══════════════════════════════════════════════════════════════════════════
# Merging pupil and saliva data
# ═══════════════════════════════════════════════════════════════════════════

def merge_pupil_saliva(
    pupil_df: pd.DataFrame,
    saliva_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join saliva τ estimates into the pupil results on participant ID.

    Saliva columns are prefixed ``saliva_`` to avoid clashes with same-named
    pupil columns.  The participant column is renamed to ``participant_id``
    (no spaces) so it can be used directly in bambi formula strings.
    """
    pupil_ids  = set(pupil_df[PUPIL_PARTICIPANT_COL].unique())
    saliva_ids = set(saliva_df[SALIVA_PARTICIPANT_COL].unique())

    saliva_only = saliva_ids - pupil_ids
    if saliva_only:
        print(
            f"  Saliva-only participants (excluded — no valid pupillometry): "
            f"{sorted(saliva_only)}"
        )

    pupil_only = pupil_ids - saliva_ids
    if pupil_only:
        print(
            f"  Pupil-only participants (no saliva τ — will be dropped from models): "
            f"{sorted(pupil_only)}"
        )

    saliva_sub = saliva_df[SALIVA_COLS_KEEP].copy().rename(
        columns={col: f"saliva_{col}" for col in SALIVA_COLS_KEEP
                 if col != SALIVA_PARTICIPANT_COL}
    )

    merged = pupil_df.merge(
        saliva_sub,
        left_on=PUPIL_PARTICIPANT_COL,
        right_on=SALIVA_PARTICIPANT_COL,
        how="left",
    )
    if SALIVA_PARTICIPANT_COL in merged.columns:
        merged = merged.drop(columns=[SALIVA_PARTICIPANT_COL])

    # Rename participant column: remove space so it is safe in formula strings
    merged = merged.rename(columns={PUPIL_PARTICIPANT_COL: ANALYSIS_PARTICIPANT_COL})

    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Data preparation
# ═══════════════════════════════════════════════════════════════════════════

def build_analysis_df(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Add analysis variables to the merged dataframe.

    New columns
    -----------
    hours_since_start : float
        Elapsed time from the start of the protocol to the beginning of each
        block.  Block n starts at n × 3.75 h.

    log_amplitude : float
        Natural log of the pupil cosinor amplitude.  Applied because raw
        amplitude is right-skewed (skew ≈ 1.24); log reduces skew to ≈ -0.60.

    alpha_sin : float
        Sine circadian regressor evaluated at each block's time, using the
        participant-specific period τ.  Its slope in the circadian model is α:
            alpha_sin = sin(2π × hours_since_start / τ_i)

    beta_cos : float
        Cosine circadian regressor, whose slope in the circadian model is β:
            beta_cos = cos(2π × hours_since_start / τ_i)
    """
    df = merged_df.copy()

    # ── Time variable ─────────────────────────────────────────────────────
    df["hours_since_start"] = df[PUPIL_BLOCK_COL] * BLOCK_INTERVAL_HOURS

    # ── Log-transform amplitudes ──────────────────────────────────────────
    df["log_amplitude"] = np.log(df["amplitude"])
    df["log_amplitude_2f"] = np.log(df["amplitude_2f"])

    # ── Baseline alias (CSV stores baseline_mean) ─────────────────────────
    df["baseline"] = df["baseline_mean"]

    # ── Phase aliases (CSV stores *_rad suffix; models use bare names) ────
    df["phase"]    = df["phase_rad"]
    df["phase_2f"] = df["phase_2f_rad"]

    # ── Per-row circadian regressors (participant-specific τ) ─────────────
    # Phase angle: 2π × t / τ_i
    phase = 2.0 * np.pi * df["hours_since_start"] / df["saliva_tau_hours"]
    df["alpha_sin"] = np.sin(phase)   # regressor carrying the α (sine) slope
    df["beta_cos"]  = np.cos(phase)   # regressor carrying the β (cosine) slope

    return df
