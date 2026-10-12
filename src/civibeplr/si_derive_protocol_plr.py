"""Derive the registered FLUX-normalised pupil outcome.

The published protocol registered the pupil endpoint

    R_ibc = Abar_ibc / Abar_ib,FLUX

where Abar is the mean *trial-level* maximum pupil constriction for participant
i, block b and condition c.  The manuscript instead analysed the fundamental
amplitude of the 0.4-Hz phasic residual and the tonic minimum.

This module derives the registered endpoint.  The order of operations is the
registered one and is load-bearing: the maximum is taken separately on every
accepted trial and only then averaged within a block.  Reversing it (averaging
trials into one trace and taking a single maximum) changes the block amplitude
by only 0.06-0.15 mm, which is small enough to pass unnoticed, so
:func:`audit_ordering` computes both and writes the difference into the
block-level output.

See ``todo/protocol_plr_supplementary_analysis_plan.md`` for the locked
decisions and the empirical characterisation of the endpoint.

The signal and the baseline window are the primary pipeline's own: pupil size
comes from ``aggregate_pupil_traces._select_pupil_size`` and the baseline is
the ``"pre-stim"`` trial phase, as in ``normalise_to_baseline``.  Only the
order of operations differs (baseline and maximum per trial, not on the
block-averaged trace).  No existing module is modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from civibeplr.analysis_io import (
    DEFAULT_T_STIM_END,
    DEFAULT_T_STIM_START,
    PUPIL_BLOCK_COL,
    PUPIL_CONDITION_COL,
    PUPIL_PARTICIPANT_COL,
)
from civibeplr.aggregate_pupil_traces import (
    DEFAULT_PHASE_COL as PHASE_COL,
    DEFAULT_TEST_COL as TEST_COL,
    DEFAULT_TIME_COL as TIME_COL,
    _select_pupil_size as select_pupil_size,
)
from civibeplr.data_io import filter_min_blocks
from civibeplr.latex_table_creation import format_latex_variables
from civibeplr.load_final_data import _iter_csvs

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_DIR = DEFAULT_PROJECT_DIR / "data" / "retinawise_results" / "final"
DEFAULT_DERIV_DIR = DEFAULT_PROJECT_DIR / "data" / "derivatives" / "plr_ratio"
DEFAULT_TEX_DIR = DEFAULT_PROJECT_DIR / "results" / "tex" / "plr_ratio"

TRIAL_NO_COL = "Trial no"

#: Identifies one pupil time series.  Every grouping in this module uses the
#: full key; dropping ``Test`` silently pools test repetitions and still
#: produces plausible amplitudes, which is why test 2 in the plan exists.
TRIAL_KEY = [
    PUPIL_PARTICIPANT_COL,
    PUPIL_BLOCK_COL,
    PUPIL_CONDITION_COL,
    TEST_COL,
    TRIAL_NO_COL,
]

BLOCK_KEY = [PUPIL_PARTICIPANT_COL, PUPIL_BLOCK_COL, PUPIL_CONDITION_COL]

FLUX_CONDITION = "flux"
NON_FLUX_CONDITIONS = ["l-m", "lms", "mel", "s"]

#: Trial phase whose samples form the baseline, as in the primary pipeline's
#: ``normalise_to_baseline``.  In the data it covers exactly [-1, 0) s, the
#: "one second prior to stimulus onset" the protocol specifies.
BASELINE_PHASE = "pre-stim"

#: The span of that phase, in seconds.  Used only for plot limits; baseline
#: samples are selected by the phase label.
BASELINE_START_SEC = -1.0
BASELINE_END_SEC = 0.0

#: Minimum distinct blocks per participant-condition series, applied *after*
#: FLUX pairing.  Matches the primary pipeline's ``filter_min_blocks`` default.
MIN_PAIRED_BLOCKS = 6

#: Sampling rate used only to bin time when auditing the reversed ordering.
SAMPLE_RATE_HZ = 30.0


def baseline_mask(df: pd.DataFrame) -> pd.Series:
    """Rows in the pre-stimulus baseline phase."""
    if PHASE_COL not in df.columns:
        raise KeyError(f"Missing required column: {PHASE_COL}")
    return df[PHASE_COL].astype(str).str.lower() == BASELINE_PHASE


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 - trial amplitudes
# ═══════════════════════════════════════════════════════════════════════════

def extract_trial_amplitudes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract one maximum-constriction amplitude per trial.

    For every trial key, the baseline is the mean pupil diameter over the
    ``"pre-stim"`` phase ([-1, 0) s) and the amplitude is
    ``baseline - min(pupil over [0, 5] s)``.
    Constriction is therefore positive.  Values at or below zero are kept as
    signed values and never converted to an absolute value.

    No smoothing is applied.  The protocol defines a maximum on the cleaned
    trace; extra filtering would define a different endpoint.

    Returns:
        One row per trial key, with baseline, stimulus minimum, sample counts
        in each window, and a validity flag.
    """
    missing = [c for c in TRIAL_KEY + [TIME_COL, PHASE_COL] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.copy()
    df["pupil_size_mm"] = select_pupil_size(df)

    time = df[TIME_COL]
    stim_mask = (time >= DEFAULT_T_STIM_START) & (time <= DEFAULT_T_STIM_END)

    baseline = (
        df[baseline_mask(df)]
        .groupby(TRIAL_KEY, dropna=False)["pupil_size_mm"]
        .agg(baseline_mm="mean", n_baseline_samples="count")
    )

    stim = df[stim_mask].dropna(subset=["pupil_size_mm"])
    stim_agg = (
        stim.groupby(TRIAL_KEY, dropna=False)["pupil_size_mm"]
        .agg(stimulus_min_mm="min", n_stimulus_samples="count")
    )

    trials = baseline.join(stim_agg, how="outer").reset_index()

    trials["constriction_amplitude_mm"] = (
        trials["baseline_mm"] - trials["stimulus_min_mm"]
    )

    trials["n_baseline_samples"] = trials["n_baseline_samples"].fillna(0).astype(int)
    trials["n_stimulus_samples"] = trials["n_stimulus_samples"].fillna(0).astype(int)

    trials["exclusion_reason"] = pd.NA
    trials.loc[trials["n_baseline_samples"] == 0, "exclusion_reason"] = "no_baseline_samples"
    trials.loc[trials["n_stimulus_samples"] == 0, "exclusion_reason"] = "no_stimulus_samples"
    trials.loc[
        trials["constriction_amplitude_mm"].isna() & trials["exclusion_reason"].isna(),
        "exclusion_reason",
    ] = "non_finite_amplitude"
    trials["trial_valid"] = trials["exclusion_reason"].isna()

    return trials


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 - block amplitudes
# ═══════════════════════════════════════════════════════════════════════════

def summarise_block_amplitudes(trials: pd.DataFrame) -> pd.DataFrame:
    """Average valid trial amplitudes within each participant-condition-block.

    Trials are pooled and weighted equally regardless of which test repetition
    they came from, matching the primary pipeline.  ``n_tests`` is retained so
    the contribution of each repetition can be audited.

    No minimum-trial rule is applied here.  The only inclusion filter in the
    primary pipeline is the six-block rule, and that is applied after FLUX
    pairing by :func:`apply_paired_block_criterion`.
    """
    valid = trials[trials["trial_valid"]].copy()

    blocks = (
        valid.groupby(BLOCK_KEY, dropna=False)
        .agg(
            n_trials=("constriction_amplitude_mm", "count"),
            n_tests=(TEST_COL, "nunique"),
            mean_constriction_mm=("constriction_amplitude_mm", "mean"),
            sd_constriction_mm=("constriction_amplitude_mm", "std"),
            min_trial_constriction_mm=("constriction_amplitude_mm", "min"),
            max_trial_constriction_mm=("constriction_amplitude_mm", "max"),
        )
        .reset_index()
    )

    blocks["exclusion_reason"] = pd.NA
    blocks.loc[blocks["n_trials"] == 0, "exclusion_reason"] = "no_valid_trials"
    blocks.loc[
        blocks["mean_constriction_mm"].isna() & blocks["exclusion_reason"].isna(),
        "exclusion_reason",
    ] = "non_finite_block_mean"
    blocks["block_valid"] = blocks["exclusion_reason"].isna()

    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 - self-audit of the order of operations
# ═══════════════════════════════════════════════════════════════════════════

def audit_ordering(df: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """Attach the reversed-ordering amplitude to the block table.

    Computes the alternative estimator -- baseline-correct each trial, average
    the trials into one trace, then take a single maximum -- and records it
    beside the registered one.  The per-trial maximum is deeper in every
    condition, by an amount that varies across conditions (the range is
    exported by ``build_derivation_summary``).  The difference mixes
    trial-to-trial response variation with measurement noise; nothing here
    separates the two.

    Writing both into the output makes the order of operations visible in the
    results rather than only in a test, and produces the numbers the supplement
    has to report anyway.
    """
    df = df.copy()
    df["pupil_size_mm"] = select_pupil_size(df)

    time = df[TIME_COL]
    baseline = (
        df[baseline_mask(df)]
        .groupby(TRIAL_KEY, dropna=False)["pupil_size_mm"]
        .mean()
        .rename("_baseline")
    )
    df = df.join(baseline, on=TRIAL_KEY)
    df["_rel"] = df["pupil_size_mm"] - df["_baseline"]

    stim = df[(time >= DEFAULT_T_STIM_START) & (time <= DEFAULT_T_STIM_END)].copy()
    stim["_time_bin"] = (stim[TIME_COL] * SAMPLE_RATE_HZ).round()

    mean_trace = stim.groupby(BLOCK_KEY + ["_time_bin"], dropna=False)["_rel"].mean()
    by_block = mean_trace.groupby(level=list(range(len(BLOCK_KEY))))
    amp_avg_trace = (-by_block.min()).rename("mean_constriction_avgtrace_mm")

    out = blocks.merge(amp_avg_trace.reset_index(), on=BLOCK_KEY, how="left")
    out["ordering_bias_mm"] = (
        out["mean_constriction_mm"] - out["mean_constriction_avgtrace_mm"]
    )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 - FLUX pairing
# ═══════════════════════════════════════════════════════════════════════════

def pair_with_flux(blocks: pd.DataFrame) -> pd.DataFrame:
    """Divide each non-FLUX block amplitude by its matched FLUX amplitude.

    The denominator is matched on participant *and* block.  It is never an
    average across blocks, another participant's FLUX, or a group mean: the
    protocol frames the comparison as isolating the "unique circadian
    variation" of each mechanism, which requires a contemporaneous denominator.

    FLUX itself is not returned, since FLUX/FLUX is 1 by construction.
    """
    valid = blocks[blocks["block_valid"]].copy()

    flux = valid[valid[PUPIL_CONDITION_COL] == FLUX_CONDITION]
    denom = flux[
        [PUPIL_PARTICIPANT_COL, PUPIL_BLOCK_COL, "mean_constriction_mm", "n_trials"]
    ].rename(
        columns={
            "mean_constriction_mm": "flux_constriction_mm",
            "n_trials": "flux_n_trials",
        }
    )

    non_flux = valid[valid[PUPIL_CONDITION_COL] != FLUX_CONDITION]

    paired = non_flux.merge(
        denom,
        on=[PUPIL_PARTICIPANT_COL, PUPIL_BLOCK_COL],
        how="left",
        validate="many_to_one",
    )

    # Denominators are strictly positive in this dataset (min 0.29 mm, 5th
    # percentile 0.43 mm).  Guard rather than branch: a zero would be a data
    # error, not a case to handle.
    zero_denom = paired["flux_constriction_mm"] == 0
    if zero_denom.any():
        raise ValueError(
            f"{int(zero_denom.sum())} block(s) have a FLUX denominator of exactly "
            "zero; the ratio is undefined. Investigate before proceeding."
        )

    paired["plr_ratio"] = (
        paired["mean_constriction_mm"] / paired["flux_constriction_mm"]
    )

    paired["pair_exclusion_reason"] = pd.NA
    paired.loc[
        paired["flux_constriction_mm"].isna(), "pair_exclusion_reason"
    ] = "no_matched_flux_block"
    paired.loc[
        paired["plr_ratio"].isna() & paired["pair_exclusion_reason"].isna(),
        "pair_exclusion_reason",
    ] = "non_finite_ratio"
    paired["pair_valid"] = paired["pair_exclusion_reason"].isna()

    return paired


def apply_paired_block_criterion(
    paired: pd.DataFrame, min_blocks: int = MIN_PAIRED_BLOCKS
) -> pd.DataFrame:
    """Drop participant-condition series with too few *paired* blocks.

    Order matters: a participant can have six valid numerator blocks but fewer
    than six with a matched FLUX denominator, so the filter runs after pairing.
    In this dataset FLUX is present in all 95 participant-blocks that carry any
    data, so the ordering changes little -- but it is the correct ordering.
    """
    analysable = paired[paired["pair_valid"]].copy()
    return filter_min_blocks(
        analysable,
        min_blocks=min_blocks,
        participant_col=PUPIL_PARTICIPANT_COL,
        condition_col=PUPIL_CONDITION_COL,
        block_col=PUPIL_BLOCK_COL,
    )



def flag_analysis_sample(
    blocks: pd.DataFrame, min_blocks: int = MIN_PAIRED_BLOCKS
) -> pd.DataFrame:
    """Add ``in_analysis_sample``: valid blocks passing the six-block rule.

    Applied per condition to valid blocks, as in the primary pipeline, so the
    flagged blocks are the main-article analysis sample (FLUX included). The
    appendix descriptives use these blocks, so they compare like with like
    against the main phasic means and the ratio models.
    """
    kept = filter_min_blocks(
        blocks[blocks["block_valid"]],
        min_blocks=min_blocks,
        participant_col=PUPIL_PARTICIPANT_COL,
        condition_col=PUPIL_CONDITION_COL,
        block_col=PUPIL_BLOCK_COL,
    )
    key = [PUPIL_PARTICIPANT_COL, PUPIL_CONDITION_COL, PUPIL_BLOCK_COL]
    kept_keys = set(map(tuple, kept[key].to_numpy()))
    out = blocks.copy()
    out["in_analysis_sample"] = [k in kept_keys for k in map(tuple, out[key].to_numpy())]
    return out

def build_exclusion_log(
    trials: pd.DataFrame,
    blocks: pd.DataFrame,
    paired: pd.DataFrame,
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Count what was lost at each stage, per condition."""
    rows: list[dict[str, object]] = []
    for cond in sorted(blocks[PUPIL_CONDITION_COL].dropna().unique()):
        t = trials[trials[PUPIL_CONDITION_COL] == cond]
        b = blocks[blocks[PUPIL_CONDITION_COL] == cond]
        p = paired[paired[PUPIL_CONDITION_COL] == cond]
        a = analysis[analysis[PUPIL_CONDITION_COL] == cond]
        rows.append(
            {
                "condition": cond,
                "n_trials_total": int(len(t)),
                "n_trials_valid": int(t["trial_valid"].sum()),
                "n_blocks_total": int(len(b)),
                "n_blocks_valid": int(b["block_valid"].sum()),
                "n_blocks_paired": int(p["pair_valid"].sum()) if len(p) else 0,
                "n_blocks_no_flux": int(
                    (p["pair_exclusion_reason"] == "no_matched_flux_block").sum()
                )
                if len(p)
                else 0,
                "n_blocks_analysis": int(len(a)),
                "n_participants_analysis": int(
                    a[PUPIL_PARTICIPANT_COL].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX variable export
# ═══════════════════════════════════════════════════════════════════════════

def cast_integral_columns(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """Return *df* with whole-number columns typed so they render without ".000".

    A count column that is NaN for one row (FLUX has no ratio, for instance)
    is coerced to float64 by pandas, and ``format_latex_variables`` would then
    write "74" as "74.000".  Columns whose non-missing values are all integral
    are converted to object dtype holding ints; NaN stays a float so the
    exporter still skips it.  ``Series.map`` cannot be used here -- it re-infers
    back to float64 as soon as one NaN is present.
    """
    out = df.copy()
    for col in out.columns:
        vals = out[col]
        if col == key_col or not pd.api.types.is_numeric_dtype(vals):
            continue
        present = vals.dropna()
        if present.empty or not (present % 1 == 0).all():
            continue  # genuinely fractional (e.g. a median of 8.5)
        out[col] = pd.Series(
            [int(v) if pd.notna(v) else float("nan") for v in vals],
            index=out.index, dtype=object,
        )
    return out


def build_derivation_summary(
    blocks: pd.DataFrame, analysis: pd.DataFrame
) -> pd.DataFrame:
    """One row per condition with the derivation quantities cited in the text.

    These are the numbers the appendix states in prose -- the endpoint by
    condition, the ordering bias, the denominator floor and the sample flow.
    Endpoint and ordering quantities use the analysis sample
    (``in_analysis_sample``, see ``flag_analysis_sample``) and are means of
    participant means, so each participant counts once whatever their number
    of retained blocks (as in the main-article summary tables and the
    appendix figures); the sample-flow counts use all valid blocks.
    The model-level quantities already have macros from ``save_results`` and
    the diagnostics; this closes the remaining gap so no number in the
    appendix is hard-coded.
    """
    rows: list[dict[str, object]] = []
    n_valid = blocks[blocks["block_valid"]].groupby(PUPIL_CONDITION_COL).size()
    sample = blocks[blocks["in_analysis_sample"]]
    for cond, g in sample.groupby(PUPIL_CONDITION_COL):
        a = analysis[analysis[PUPIL_CONDITION_COL] == cond]
        pp = g.groupby(PUPIL_PARTICIPANT_COL)[
            ["ordering_bias_mm", "mean_constriction_mm", "mean_constriction_avgtrace_mm"]
        ].mean()
        bias = pp["ordering_bias_mm"].mean()
        endpoint = pp["mean_constriction_mm"].mean()
        row: dict[str, object] = {
            "condition": cond,
            "endpoint_mm": endpoint,
            "endpoint_avgtrace_mm": pp["mean_constriction_avgtrace_mm"].mean(),
            "ordering_bias_mm": bias,
            "ordering_bias_pct": 100.0 * bias / endpoint if endpoint else np.nan,
            "n_blocks_valid": int(n_valid[cond]),
            "n_blocks_sample": int(len(g)),
            "n_trials_median": float(g["n_trials"].median()),
        }
        if not a.empty:
            row.update(
                {
                    "n_obs": int(len(a)),
                    "n_participants": int(a[PUPIL_PARTICIPANT_COL].nunique()),
                    "ratio_mean": float(a["plr_ratio"].mean()),
                    "ratio_sd": float(a["plr_ratio"].std()),
                    "ratio_median": float(a["plr_ratio"].median()),
                    "flux_denominator_min_mm": float(a["flux_constriction_mm"].min()),
                }
            )
        rows.append(row)

    out = pd.DataFrame(rows)

    # Cross-condition scalars, so ranges quoted in prose are data-driven too.
    modelled = out[out["condition"] != FLUX_CONDITION]
    piv_pt = sample.pivot_table(
        index=[PUPIL_PARTICIPANT_COL, PUPIL_BLOCK_COL],
        columns=PUPIL_CONDITION_COL, values="mean_constriction_mm")
    piv_av = sample.pivot_table(
        index=[PUPIL_PARTICIPANT_COL, PUPIL_BLOCK_COL],
        columns=PUPIL_CONDITION_COL, values="mean_constriction_avgtrace_mm")
    corrs, within_pt, within_av = [], [], []
    for cond in NON_FLUX_CONDITIONS:
        if cond not in piv_pt.columns:
            continue
        r_pt = (piv_pt[cond] / piv_pt[FLUX_CONDITION]).dropna()
        r_av = (piv_av[cond] / piv_av[FLUX_CONDITION]).dropna()
        a, b = r_pt.align(r_av, join="inner")
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
        within_pt.append(float(r_pt.groupby(level=0).std().mean()))
        within_av.append(float(r_av.groupby(level=0).std().mean()))

    all_row = {
        "condition": "all",
        "ordering_bias_mm_min": out["ordering_bias_mm"].min(),
        "ordering_bias_mm_max": out["ordering_bias_mm"].max(),
        "ordering_bias_pct_min": out["ordering_bias_pct"].min(),
        "ordering_bias_pct_max": out["ordering_bias_pct"].max(),
        "ordering_bias_fold": out["ordering_bias_mm"].max() / out["ordering_bias_mm"].min(),
        "ordering_corr_min": min(corrs) if corrs else np.nan,
        "ordering_corr_max": max(corrs) if corrs else np.nan,
        "within_pp_sd_pertrial_min": min(within_pt) if within_pt else np.nan,
        "within_pp_sd_pertrial_max": max(within_pt) if within_pt else np.nan,
        "within_pp_sd_avgtrace_min": min(within_av) if within_av else np.nan,
        "within_pp_sd_avgtrace_max": max(within_av) if within_av else np.nan,
        # Cast counts so they render as "74" rather than "74.000".
        "n_obs_min": int(modelled["n_obs"].min()),
        "n_obs_max": int(modelled["n_obs"].max()),
        "n_participants_max": int(modelled["n_participants"].max()),
        "n_participants_min": int(modelled["n_participants"].min()),
        "n_blocks_any_data": int(
            blocks.loc[blocks[PUPIL_CONDITION_COL] == FLUX_CONDITION, "block_valid"].sum()
        ),
        "flux_denominator_min_mm": modelled["flux_denominator_min_mm"].min(),
    }
    return pd.concat([out, pd.DataFrame([all_row])], ignore_index=True)


def export_latex_variables(
    summary: pd.DataFrame, tex_dir: Path | None = None
) -> Path:
    """Write ``vars_plr_ratio.tex`` with one \def per non-NaN cell.

    Prefix ``PlrRatio`` keeps these distinct from the ``CircadianCosinor``
    and ``DiagnoseCircadian`` macros written by the modelling step.
    """
    tex_dir = tex_dir or DEFAULT_TEX_DIR
    tex_dir.mkdir(parents=True, exist_ok=True)

    # Count columns hold NaN for FLUX (which has no ratio), which coerces the
    # whole column to float and would render "74" as "74.000".  Map to object
    # dtype holding ints, keeping NaN as a float so the exporter still skips it.
    summary = cast_integral_columns(summary, key_col="condition")

    tex = format_latex_variables(
        summary, row_key_col="condition", fmt=".3f", prefix="plr_ratio"
    )
    out = tex_dir / "vars_plr_ratio.tex"
    out.write_text(tex, encoding="utf-8")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Assertions
# ═══════════════════════════════════════════════════════════════════════════

def assert_derivation_sane(trials: pd.DataFrame, analysis: pd.DataFrame) -> None:
    """Runtime checks for situations the plan says must not arise."""
    dupes = trials.duplicated(subset=TRIAL_KEY).sum()
    if dupes:
        raise AssertionError(f"{dupes} duplicate trial keys; key is not unique.")

    unexpected = set(analysis[PUPIL_CONDITION_COL].unique()) - set(NON_FLUX_CONDITIONS)
    if unexpected:
        raise AssertionError(f"Unexpected conditions in analysis set: {sorted(unexpected)}")

    if FLUX_CONDITION in set(analysis[PUPIL_CONDITION_COL].unique()):
        raise AssertionError("FLUX must not appear in the modelled condition set.")

    if not np.isfinite(analysis["plr_ratio"]).all():
        raise AssertionError("Non-finite plr_ratio in the analysis set.")

    n_nonpositive = int((analysis["mean_constriction_mm"] <= 0).sum())
    if n_nonpositive:
        # Not fatal, but the plan requires it to be surfaced rather than hidden.
        print(
            f"  WARNING: {n_nonpositive} analysis block(s) have a non-positive "
            "constriction amplitude; these are kept signed and must be reported."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Orchestration
# ═══════════════════════════════════════════════════════════════════════════

def derive_plr_ratio_endpoint(
    final_dir: Path | None = None,
    output_dir: Path | None = None,
    write: bool = True,
) -> dict[str, pd.DataFrame]:
    """Run the full derivation and optionally persist each intermediate level.

    Data accessed:
        The same per-participant files the primary pipeline starts from:
        ``data/retinawise_results/final/<ID>_final_data.csv`` (output of the
        external civibe_pupilprep repo), matched by ``[0-9][0-9][0-9]_*.csv``.
        It does *not* read ``data/aggregated/data.csv`` or any cosinor output
        from Steps 1-5; the endpoint is derived independently from the
        sample-level traces.  Each file is loaded and processed on its own to
        keep memory bounded, then the trial and block tables are concatenated.

    Processing, compared with the primary pipeline:
        Shared with the primary pipeline (Step 1, ``aggregate_pupil_traces``):

        - pupil size from the measured eye (``_select_pupil_size``);
        - baseline = mean over the ``"pre-stim"`` phase ([-1, 0) s);
        - stimulus window [DEFAULT_T_STIM_START, DEFAULT_T_STIM_END] (0-5 s);
        - test repetitions pooled with equal weight within a block;
        - the six-block rule (``filter_min_blocks``).

        Different from the primary pipeline:

        - Unit of aggregation.  Step 1 averages every sample at each time
          point across trials and tests into one trace per
          participant-condition-block, then baseline-corrects that trace.
          Here the baseline and the maximum constriction are computed on each
          trial (``TRIAL_KEY``) first and only then averaged within the block.
        - Outcome.  Steps 2-5 fit a 0.4-Hz cosinor to the block-averaged
          ``rel_size_mm`` trace (after a 0.2-Hz low-pass split into tonic and
          phasic parts) and model its amplitude and tonic minimum.  Here the
          outcome is a scalar per block, ``baseline - min(pupil, 0-5 s)``, on
          the unfiltered trace, with no cosinor fit and no filtering.
        - Normalisation.  Each non-FLUX block amplitude is divided by the
          FLUX amplitude of the same participant and block (``plr_ratio``).
          The primary pipeline models each condition on its own scale.
        - Six-block rule timing.  Step 1 applies it to the aggregated traces;
          here it is applied after FLUX pairing, to blocks that have a valid
          ratio.  FLUX itself is dropped from the analysis set.
        - Ordering audit.  The reversed (primary-pipeline-style) estimator,
          max of the trial-averaged trace, is computed alongside and stored as
          ``mean_constriction_avgtrace_mm`` / ``ordering_bias_mm``.

    Args:
        final_dir: Directory with the per-participant final CSVs.  Defaults
            to ``DEFAULT_FINAL_DIR``.
        output_dir: Directory for the derived CSVs.  Defaults to
            ``data/derivatives/plr_ratio/``.
        write: If True, write the six CSVs below to *output_dir* and the
            LaTeX macros to ``results/tex/plr_ratio/vars_plr_ratio.tex``
            (always ``DEFAULT_TEX_DIR``, independent of *output_dir*).

    Returns:
        Dict of DataFrames, each also written as a CSV when *write* is True:

        - ``trials`` (``trial_amplitudes.csv``): one row per trial, with
          baseline, stimulus minimum and validity flag.
        - ``blocks`` (``block_amplitudes.csv``): one row per
          participant-condition-block, including the ordering audit and the
          ``in_analysis_sample`` flag used by the appendix descriptives.
        - ``paired`` (``paired_ratios_all.csv``): non-FLUX blocks with their
          matched FLUX denominator and ``plr_ratio``.
        - ``analysis`` (``paired_ratios_analysis.csv``): valid pairs passing
          the six-block rule; the input to ``si_analyse_plr_ratio`` and
          ``si_sensitivity_protocol_plr``.
        - ``exclusions`` (``exclusion_log.csv``): counts lost at each stage.
        - ``summary`` (``derivation_summary.csv``): per-condition quantities
          cited in the appendix, exported as ``\\PlrRatio...`` macros.
    """
    final_dir = final_dir or DEFAULT_FINAL_DIR
    output_dir = output_dir or DEFAULT_DERIV_DIR

    csv_paths = list(_iter_csvs(final_dir, "[0-9][0-9][0-9]_*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No participant CSVs found in: {final_dir}")

    print(f"Deriving protocol PLR endpoint from {len(csv_paths)} participant files...")

    trial_frames: list[pd.DataFrame] = []
    block_frames: list[pd.DataFrame] = []

    for path in csv_paths:
        raw = pd.read_csv(path, index_col=0)
        trials = extract_trial_amplitudes(raw)
        blocks = summarise_block_amplitudes(trials)
        blocks = audit_ordering(raw, blocks)
        trial_frames.append(trials)
        block_frames.append(blocks)

    trials = pd.concat(trial_frames, ignore_index=True)
    blocks = pd.concat(block_frames, ignore_index=True)

    print(f"  Trials: {len(trials)} ({int(trials['trial_valid'].sum())} valid)")
    print(f"  Blocks: {len(blocks)} ({int(blocks['block_valid'].sum())} valid)")
    blocks = flag_analysis_sample(blocks)
    print(f"  Analysis sample: {int(blocks['in_analysis_sample'].sum())} blocks")

    print("\nPairing each condition with its matched FLUX block...")
    paired = pair_with_flux(blocks)
    print(f"  Paired: {int(paired['pair_valid'].sum())} of {len(paired)} non-FLUX blocks")

    print(f"\nApplying the {MIN_PAIRED_BLOCKS}-block rule after pairing...")
    analysis = apply_paired_block_criterion(paired)

    assert_derivation_sane(trials, analysis)

    exclusions = build_exclusion_log(trials, blocks, paired, analysis)
    summary = build_derivation_summary(blocks, analysis)

    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        trials.to_csv(output_dir / "trial_amplitudes.csv", index=False)
        blocks.to_csv(output_dir / "block_amplitudes.csv", index=False)
        paired.to_csv(output_dir / "paired_ratios_all.csv", index=False)
        analysis.to_csv(output_dir / "paired_ratios_analysis.csv", index=False)
        exclusions.to_csv(output_dir / "exclusion_log.csv", index=False)
        summary.to_csv(output_dir / "derivation_summary.csv", index=False)
        print(f"\nWrote 6 files to {output_dir}")
        tex_path = export_latex_variables(summary)
        print(f"Wrote LaTeX variables: {tex_path}")

    return {
        "trials": trials,
        "blocks": blocks,
        "paired": paired,
        "analysis": analysis,
        "exclusions": exclusions,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive the registered FLUX-normalised pupil outcome."
    )
    parser.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DERIV_DIR)
    parser.add_argument("--no-write", action="store_true", help="Derive without saving.")
    args = parser.parse_args()

    out = derive_plr_ratio_endpoint(
        final_dir=args.final_dir,
        output_dir=args.output_dir,
        write=not args.no_write,
    )

    analysis = out["analysis"]
    blocks = out["blocks"]

    print("\n── Exclusion flow ──")
    print(out["exclusions"].to_string(index=False))

    print("\n── Ordering audit (mm) ──")
    print(
        blocks.groupby(PUPIL_CONDITION_COL)[
            ["mean_constriction_mm", "mean_constriction_avgtrace_mm", "ordering_bias_mm"]
        ]
        .mean()
        .round(3)
        .to_string()
    )

    print("\n── Ratio by condition ──")
    print(
        analysis.groupby(PUPIL_CONDITION_COL)["plr_ratio"]
        .agg(["count", "mean", "std", "median", "min", "max", "skew"])
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
