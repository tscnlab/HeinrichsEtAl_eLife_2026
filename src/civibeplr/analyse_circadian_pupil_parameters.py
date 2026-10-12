"""analyze_circadian_pupil_parameters.py

Stage 2 analysis: test whether the log-amplitude of the pupil cosinor
(from the two-step pipeline) follows a near-circadian rhythm across blocks,
using participant-specific τ estimates from the saliva cosinor model.

Model specification (one model fitted per condition)
----------------------------------------------------
Outcome : amplitude   — of fundamental frequency component (τ_i) from the two-step cosinor pipeline

Time    : hours_since_start = Block × 3.75 h  (blocks are 3 h 45 min apart)

Regressors (participant-specific τ), named after the slope they carry:
    alpha_sin = sin(2π × hours_since_start / τ_i)   -> slope α
    beta_cos  = cos(2π × hours_since_start / τ_i)   -> slope β

M0 (null)    : pupil_amplitude ~ 1 + (1 | participant_id)
M1 (cosinor) : pupil_amplitude ~ alpha_sin + beta_cos + (1 + alpha_sin + beta_cos | participant_id)

Both models are fitted with pymc.sample_smc() to obtain the log marginal
likelihood, which is then used to compute the Bayes factor BF₁₀.

Packages: bambi (model specification), pymc (SMC sampling)
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import bambi as bmb

from civibeplr.analysis_io import (
    ANALYSIS_PARTICIPANT_COL,
    PARAMETER_CHOICES,
    PUPIL_BLOCK_COL,
    PUPIL_CONDITION_COL,
    PUPIL_PARTICIPANT_COL,
    build_analysis_df,
    load_pupil_results,
    load_saliva_results,
    merge_pupil_saliva,
)
from civibeplr.inspect_circadian_model import inspect_parameter_distribution
from civibeplr.latex_table_creation import (
    format_condition_sample_sizes_latex,
    format_latex_variables,
    format_circadian_parameter_estimates_latex,
    format_circadian_model_summary_latex,
    format_circadian_model_summary_all_metrics_latex,
)
from civibeplr.utils_cosinor import (
    backtransform_log_amplitude_circadian,
    circ_interval,
    circ_mean,
    circ_sd,
)

# ── Default paths ────────────────────────────────────────────────────────────

_PROJECT_DIR = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _PROJECT_DIR / "results" / "tables"

DEFAULT_OUTPUT_DIR = _RESULTS_DIR / "circadian_cosinor"
DEFAULT_OUTPUT_TEX_DIR = _PROJECT_DIR / "results" / "tex" / "circadian_cosinor"

# ── Prior hyperparameters (dimensionless scale factors) ──────────────────────
# Priors are expressed as multiples of the outcome's empirical SD so that
# the same prior specification remains sensible for any outcome metric
# (log-amplitude, raw amplitude, tonic minimum, phase, etc.).
#
# intercept σ = PRIOR_INTERCEPT_SD_FACTOR × SD(outcome)
#   Factor of 1.5 sets the prior to ≈1.5× the data spread — weakly
#   informative but not data-dominated.
#
# slope σ = PRIOR_SLOPE_SD_FACTOR × SD(outcome)
#   For log-amplitude (SD ≈ 0.70), factor 0.7 → σ ≈ 0.49, reproducing the
#   previous hard-coded 0.5.  For a different metric the prior automatically
#   scales with the outcome's natural unit.

PRIOR_INTERCEPT_SD_FACTOR = 1.5  # intercept prior σ = factor × SD(outcome)
PRIOR_SLOPE_SD_FACTOR = 0.7  # slope prior σ     = factor × SD(outcome)

# ── SMC sampling ─────────────────────────────────────────────────────────────
# Number of SMC particles per model.  The Bayes factor is derived from the SMC
# estimate of the log marginal likelihood (log Z), which is biased away from
# zero when too few particles are used; the bias shrinks as the particle count
# grows.  At 2000 draws the log BF10 estimates were still drifting substantially with increased number of draws, so the default was
# raised to 10000.  Note that the per-chain log Z SD reported by the diagnostics
# measures scatter *within* a run and does not detect this particle-count bias.
DEFAULT_SMC_DRAWS = 10000
# Number of independent SMC particle populations (PyMC "chains").  Pinned
# because PyMC otherwise derives it from the CPU count; the reported fits used 5.
DEFAULT_SMC_CHAINS = 5

# ── Output paths for InferenceData and figures ────────────────────────────────

DEFAULT_IDATA_DIR = _PROJECT_DIR / "data" / "derivatives" / "circadian_cosinor"


# ═══════════════════════════════════════════════════════════════════════════
# Model building
# ═══════════════════════════════════════════════════════════════════════════

def build_models(
    df_cond: pd.DataFrame,
    outcome_col: str = "amplitude",
) -> tuple[bmb.Model, bmb.Model]:
    """Build M0 (null) and M1 (cosinor) bambi models for one condition.

    Models are *defined and compiled* (``model.build()`` is called) but
    NOT sampled here.  Call ``fit_condition()`` to run SMC sampling.

    M0 — intercept + random intercept
    ----------------------------------
        outcome ~ 1 + (1 | participant_id)

    M1 — cosinor fixed effects + random intercept and random slopes
    ---------------------------------------------------------------
        outcome ~ alpha_sin + beta_cos + (1 + alpha_sin + beta_cos | participant_id)

    where alpha_sin = sin(2π t / τ_i) carries the α slope and
    beta_cos = cos(2π t / τ_i) carries the β slope.

    Prior specification
    -------------------
    Intercept : Normal(μ = empirical mean of outcome, σ = empirical SD of outcome × PRIOR_INTERCEPT_SD_FACTOR)
    alpha_sin, beta_cos : Normal(0, empirical SD of outcome × PRIOR_SLOPE_SD_FACTOR)
    Random effect SDs : HalfStudentT(ν=4, σ = scale) hyperprior, where scale = outcome_sd
        for the intercept and scale = slope_sigma for the slopes.  HalfStudentT is preferred
        over HalfNormal because its heavier tails accommodate participants with unusually
        strong or weak circadian rhythms without over-shrinking.  Scaling by the empirical
        data spread keeps the prior weakly informative regardless of the outcome metric.
        Bambi 0.17 treats random effects as independent (diagonal covariance); no correlation
        matrix is estimated.
    Residual σ : bambi default, HalfStudentT(ν=4, σ = SD(outcome)); not set here
    """
    intercept_mu = float(df_cond[outcome_col].mean())
    outcome_sd   = float(df_cond[outcome_col].std())

    intercept_sigma = outcome_sd * PRIOR_INTERCEPT_SD_FACTOR
    slope_sigma     = outcome_sd * PRIOR_SLOPE_SD_FACTOR

    re_int_sd_prior   = bmb.Prior("Normal", sigma=bmb.Prior("HalfStudentT", nu=4, sigma=outcome_sd))
    re_slope_sd_prior = bmb.Prior("Normal", sigma=bmb.Prior("HalfStudentT", nu=4, sigma=slope_sigma))

    priors_m0 = {
        "Intercept":                         bmb.Prior("Normal", mu=intercept_mu, sigma=intercept_sigma),
        f"1|{ANALYSIS_PARTICIPANT_COL}":     re_int_sd_prior,
    }
    priors_m1 = {
        "Intercept":                             bmb.Prior("Normal", mu=intercept_mu, sigma=intercept_sigma),
        "alpha_sin":                             bmb.Prior("Normal", mu=0.0, sigma=slope_sigma),
        "beta_cos":                              bmb.Prior("Normal", mu=0.0, sigma=slope_sigma),
        f"1|{ANALYSIS_PARTICIPANT_COL}":         re_int_sd_prior,
        f"alpha_sin|{ANALYSIS_PARTICIPANT_COL}": re_slope_sd_prior,
        f"beta_cos|{ANALYSIS_PARTICIPANT_COL}":  re_slope_sd_prior,
    }

    m0 = bmb.Model(
        formula=f"{outcome_col} ~ 1 + (1 | {ANALYSIS_PARTICIPANT_COL})",
        data=df_cond,
        priors=priors_m0,
        family="gaussian",
    )

    m1 = bmb.Model(
        formula=(f"{outcome_col} ~ alpha_sin + beta_cos"
                 f" + (1 + alpha_sin + beta_cos | {ANALYSIS_PARTICIPANT_COL})"),
        data=df_cond,
        priors=priors_m1,
        family="gaussian",
    )

    # Compile the PyMC graph without sampling so the structure can be inspected
    m0.build()
    m1.build()

    return m0, m1


# ═══════════════════════════════════════════════════════════════════════════
# Fitting
# ═══════════════════════════════════════════════════════════════════════════

def fit_condition(
    m0: bmb.Model,
    m1: bmb.Model,
    smc_draws: int = DEFAULT_SMC_DRAWS,
    random_seed: int = 42,
) -> tuple:
    """Run SMC sampling on M0 and M1 and return both InferenceData objects.

    PyMC's Sequential Monte Carlo (SMC) computes the log marginal likelihood
    (log Z).  It is stored in
        idata.sample_stats["log_marginal_likelihood"]
    and is used to compute BF10 = Z_M1 / Z_M0.
    """
    import pymc as pm

    print("  Sampling M0 (null)...")
    with m0.backend.model:
        idata_m0 = pm.sample_smc(
            draws=smc_draws, chains=DEFAULT_SMC_CHAINS, random_seed=random_seed
        )

    print("  Sampling M1 (cosinor)...")
    with m1.backend.model:
        idata_m1 = pm.sample_smc(
            draws=smc_draws, chains=DEFAULT_SMC_CHAINS, random_seed=random_seed
        )

    print("  Sampling prior predictive (M1)...")
    with m1.backend.model:
        idata_prior = pm.sample_prior_predictive(random_seed=random_seed)
    idata_m1.extend(idata_prior)

    return idata_m0, idata_m1


def compute_rhat_summary(idata_m0, idata_m1) -> tuple[pd.DataFrame, float, float]:
    """Compute per-parameter r_hat for M0 and M1.

    Each SMC "chain" here is an independently-initialized particle population
    (own prior draws, own resampling/mutation history), not a split of one
    sequential MCMC chain, so r_hat is comparing independent posterior
    approximations rather than checking within-chain mixing. It can catch
    multimodality, particle degeneracy, or seed-dependent bias between runs,
    but -- unlike MCMC r_hat -- it says nothing about how well any single SMC
    run explored the posterior, since the final-stage particles within one
    run are already close to i.i.d.

    Returns
    -------
    df : DataFrame with columns [model, parameter, rhat]
    max_rhat_m0 : float
    max_rhat_m1 : float
    """
    import arviz as az

    rows = []
    max_vals = {}
    for label, idata in [("M0", idata_m0), ("M1", idata_m1)]:
        rhat_ds = az.rhat(idata)
        values = []
        for var in rhat_ds.data_vars:
            arr = rhat_ds[var].values.flatten()
            for v in arr[~np.isnan(arr)]:
                rows.append({"model": label, "parameter": var, "rhat": float(v)})
                values.append(float(v))
        max_vals[label] = float(max(values)) if values else np.nan

    df = pd.DataFrame(rows)
    return df, max_vals["M0"], max_vals["M1"]


# ═══════════════════════════════════════════════════════════════════════════
# InferenceData persistence
# ═══════════════════════════════════════════════════════════════════════════

def load_idata(
    condition: str,
    idata_dir: Path | None = None,
):
    """Load M0 and M1 InferenceData from NetCDF files written by save_idata().

    Returns (idata_m0, idata_m1), or (None, None) if either file is missing.
    """
    import arviz as az

    if idata_dir is None:
        idata_dir = DEFAULT_IDATA_DIR
    cond_safe = condition.replace(" ", "_").lower()
    path_m0 = idata_dir / f"{cond_safe}_M0.nc"
    path_m1 = idata_dir / f"{cond_safe}_M1.nc"
    if not path_m0.exists() or not path_m1.exists():
        return None, None
    idata_m0 = az.from_netcdf(str(path_m0))
    idata_m1 = az.from_netcdf(str(path_m1))
    print(f"  Loaded: {path_m0.name}, {path_m1.name}")
    return idata_m0, idata_m1


def save_idata(
    m1: bmb.Model,
    idata_m0,
    idata_m1,
    condition: str,
    outcome_col: str = "amplitude",
    idata_dir: Path | None = None,
) -> None:
    """Save M0 and M1 InferenceData as NetCDF files and the training data as CSV.

    Before saving, conditional posterior predictions (mu) are added in-place to
    idata_m1.posterior via ``model.predict()``. This allows downstream
    diagnostic scripts to compute residuals without rebuilding the model.

    The training DataFrame (``m1.data``) is also saved so that the diagnose
    script can load observations that are guaranteed to align with the mu
    predictions stored in idata_m1.

    Files written
    -------------
    {idata_dir}/{cond_safe}_M0.nc
    {idata_dir}/{cond_safe}_M1.nc            (enriched with conditional mu)
    {idata_dir}/{cond_safe}_data.csv         (training data used to build M1)
    {idata_dir}/{cond_safe}_predictions.csv  (per-observation conditional + marginal mu)
    {idata_dir}/{cond_safe}_blups.csv        (per-participant BLUP-equivalent coefficients)
    {idata_dir}/{cond_safe}_rhat.csv         (per-parameter r_hat for M0 and M1)
    """
    if idata_dir is None:
        idata_dir = DEFAULT_IDATA_DIR
    idata_dir.mkdir(parents=True, exist_ok=True)

    # Add conditional mu predictions to idata_m1 in-place for diagnostic reuse
    m1.predict(idata_m1, kind="response_params", include_group_specific=True, inplace=True)
    cond_safe = condition.replace(" ", "_").lower()
    _sanitize_idata_for_netcdf(idata_m0)
    _sanitize_idata_for_netcdf(idata_m1)
    idata_m0.to_netcdf(str(idata_dir / f"{cond_safe}_M0.nc"))
    idata_m1.to_netcdf(str(idata_dir / f"{cond_safe}_M1.nc"))
    m1.data.to_csv(idata_dir / f"{cond_safe}_data.csv", index=False)
    print(f"  Saved InferenceData + data: {idata_dir / cond_safe}_{{M0,M1}}.nc / _data.csv")

    rhat_df, max_rhat_m0, max_rhat_m1 = compute_rhat_summary(idata_m0, idata_m1)
    rhat_path = idata_dir / f"{cond_safe}_rhat.csv"
    rhat_df.to_csv(rhat_path, index=False)
    flag_m0 = " [!]" if max_rhat_m0 > 1.01 else ""
    flag_m1 = " [!]" if max_rhat_m1 > 1.01 else ""
    print(f"  r_hat  M0 max={max_rhat_m0:.4f}{flag_m0}  M1 max={max_rhat_m1:.4f}{flag_m1}  -> {rhat_path.name}")

    save_predictions_and_blups(m1, idata_m1, condition, outcome_col=outcome_col, idata_dir=idata_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Per-observation predictions and BLUP-equivalent participant coefficients
# ═══════════════════════════════════════════════════════════════════════════

def save_predictions_and_blups(
    m1: bmb.Model,
    idata_m1,
    condition: str,
    outcome_col: str = "amplitude",
    idata_dir: Path | None = None,
) -> None:
    """Save per-observation predictions and per-participant BLUP-equivalent coefficients.

    Prediction columns (conditional = fixed + random; marginal = fixed only):
        participant_id, block, hours_since_start, {outcome_col},
        mu_cond_mean, mu_cond_lo, mu_cond_hi,
        mu_marg_mean, mu_marg_lo, mu_marg_hi

    BLUP columns (posterior mean and 95 % HDI per participant):
        participant_id,
        intercept_mean/lo/hi, alpha_mean/lo/hi, beta_mean/lo/hi,
        amplitude_mean/lo/hi, phase_mean/lo/hi
    """
    import copy

    if idata_dir is None:
        idata_dir = DEFAULT_IDATA_DIR
    cond_safe = condition.replace(" ", "_").lower()

    df = m1.data.reset_index(drop=True)

    # ── Conditional mu (fixed + random; already in posterior from save_idata) ──
    mu_cond      = np.array(idata_m1.posterior["mu"])  # (chains, draws, n_obs)
    mu_cond_mean = mu_cond.mean(axis=(0, 1))
    mu_cond_lo   = np.percentile(mu_cond, 2.5,  axis=(0, 1))
    mu_cond_hi   = np.percentile(mu_cond, 97.5, axis=(0, 1))

    # ── Marginal mu (fixed effects only) ─────────────────────────────────────
    idata_marg = copy.deepcopy(idata_m1)
    m1.predict(idata_marg, kind="response_params",
               include_group_specific=False, inplace=True)
    mu_marg      = np.array(idata_marg.posterior["mu"])
    mu_marg_mean = mu_marg.mean(axis=(0, 1))
    mu_marg_lo   = np.percentile(mu_marg, 2.5,  axis=(0, 1))
    mu_marg_hi   = np.percentile(mu_marg, 97.5, axis=(0, 1))
    del idata_marg

    # ── Predictions CSV (one row per observation) ─────────────────────────────
    keep_cols = [ANALYSIS_PARTICIPANT_COL, PUPIL_BLOCK_COL]
    if "hours_since_start" in df.columns:
        keep_cols.append("hours_since_start")
    preds_df = df[keep_cols].copy()
    preds_df[outcome_col]    = df[outcome_col].values
    preds_df["mu_cond_mean"] = mu_cond_mean
    preds_df["mu_cond_lo"]   = mu_cond_lo
    preds_df["mu_cond_hi"]   = mu_cond_hi
    preds_df["mu_marg_mean"] = mu_marg_mean
    preds_df["mu_marg_lo"]   = mu_marg_lo
    preds_df["mu_marg_hi"]   = mu_marg_hi

    preds_path = idata_dir / f"{cond_safe}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)
    print(f"  Saved predictions: {preds_path.name}")

    # ── BLUP-equivalent participant coefficients ──────────────────────────────
    post = idata_m1.posterior

    # Fixed effects: (chains, draws)
    fe_intercept = np.array(post["Intercept"])
    fe_alpha_sin = np.array(post["alpha_sin"])
    fe_beta_cos  = np.array(post["beta_cos"])

    # Random effects: (chains, draws, n_participants)
    re_int_da          = post["1|participant_id"]
    re_alpha_sin_da    = post["alpha_sin|participant_id"]
    re_beta_cos_da     = post["beta_cos|participant_id"]

    # Participant IDs from the last coordinate dimension
    part_dim  = re_int_da.dims[-1]
    part_ids  = re_int_da.coords[part_dim].values

    re_intercept = np.array(re_int_da)
    re_alpha_sin = np.array(re_alpha_sin_da)
    re_beta_cos  = np.array(re_beta_cos_da)

    # Participant-specific totals: (chains, draws, n_participants)
    intercept_p  = fe_intercept[..., None] + re_intercept
    alpha_sin_p  = fe_alpha_sin[..., None] + re_alpha_sin
    beta_cos_p   = fe_beta_cos[..., None]  + re_beta_cos
    amplitude_p  = np.sqrt(alpha_sin_p ** 2 + beta_cos_p ** 2)
    # Convention y = μ + A·cos(ωt + φ), so φ = atan2(-α, β) and the peak
    # occurs at t = -φ/ω.  Matches the pupil and saliva cosinors.
    phase_p      = np.arctan2(-alpha_sin_p, beta_cos_p)

    def _ci95(arr_3d):
        flat = arr_3d.reshape(-1, arr_3d.shape[-1])
        return flat.mean(axis=0), np.percentile(flat, 2.5, axis=0), np.percentile(flat, 97.5, axis=0)

    def _circ_ci95(arr_3d):
        flat = arr_3d.reshape(-1, arr_3d.shape[-1])
        mean = np.array([circ_mean(col) for col in flat.T])
        lo, hi = np.array([circ_interval(col, 0.95) for col in flat.T]).T
        return mean, lo, hi

    int_mean, int_lo, int_hi                      = _ci95(intercept_p)
    alpha_sin_mean, alpha_sin_lo, alpha_sin_hi    = _ci95(alpha_sin_p)
    beta_cos_mean, beta_cos_lo, beta_cos_hi       = _ci95(beta_cos_p)
    amp_mean, amp_lo, amp_hi                      = _ci95(amplitude_p)
    # Phase is an angle: circular mean, interval unwrapped around it
    pha_mean, pha_lo, pha_hi                      = _circ_ci95(phase_p)

    blups_df = pd.DataFrame({
        "participant_id":  part_ids,
        "intercept_mean":  int_mean,
        "intercept_lo":    int_lo,
        "intercept_hi":    int_hi,
        "alpha_mean":      alpha_sin_mean,
        "alpha_lo":        alpha_sin_lo,
        "alpha_hi":        alpha_sin_hi,
        "beta_mean":       beta_cos_mean,
        "beta_lo":         beta_cos_lo,
        "beta_hi":         beta_cos_hi,
        "amplitude_mean":  amp_mean,
        "amplitude_lo":    amp_lo,
        "amplitude_hi":    amp_hi,
        "phase_mean":      pha_mean,
        "phase_lo":        pha_lo,
        "phase_hi":        pha_hi,
    })
    blups_path = idata_dir / f"{cond_safe}_blups.csv"
    blups_df.to_csv(blups_path, index=False)
    print(f"  Saved BLUPs: {blups_path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Results extraction helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize_idata_for_netcdf(idata) -> None:
    """Make an InferenceData safe to serialise as NetCDF.

    Two passes:

    1. ``log_marginal_likelihood`` — converted from an object array of per-stage
       history lists to a plain ``(chain,)`` float array holding the final
       converged log Z per chain.  This preserves the value needed for BF
       computation while removing the unserializable object dtype.

    2. All remaining object-dtype variables in every InferenceData group are
       dropped.  PyMC SMC can produce additional object-array variables in
       ``sample_stats`` (e.g. per-stage tuning info) that NetCDF cannot handle.
       These are not used downstream so dropping them is safe.
    """
    import xarray as xr

    # ── Pass 1: fix log_marginal_likelihood ───────────────────────────────
    ss = idata.sample_stats
    if "log_marginal_likelihood" in ss:
        raw   = np.array(ss["log_marginal_likelihood"])
        first = raw.flat[0]
        if isinstance(first, (list, np.ndarray)):
            # List layout: each cell holds one chain's per-stage history, but the
            # cells lie along `chain` or along `draw` depending on how PyMC ran
            # the chains. Flatten so no chain is dropped, matching the layout
            # handling in _log_marginal_likelihood. 
            chain_lmls = np.array([float(cell[-1]) for cell in raw.flatten()])
        else:
            # Scalar layout: rows are chains, columns stages; take the last stage.
            chain_lmls = np.array(np.array(raw, dtype=float)[:, -1])

        if len(chain_lmls) == ss.sizes.get("chain", len(chain_lmls)):
            idata.sample_stats["log_marginal_likelihood"] = xr.DataArray(
                chain_lmls,
                dims=["chain"],
                coords={"chain": ss.coords["chain"].values},
            )
        else:
            # Cells were laid out along `draw`, so the recovered values outnumber
            # the group's `chain` dimension. Give them their own dimension rather
            # than truncating to fit it.
            idata.sample_stats = ss.drop_vars("log_marginal_likelihood").assign(
                log_marginal_likelihood=xr.DataArray(
                    chain_lmls,
                    dims=["smc_chain"],
                    coords={"smc_chain": np.arange(len(chain_lmls))},
                )
            )

    # ── Pass 2: drop any remaining object-dtype variables in all groups ───
    for group_name in idata._groups:
        group = getattr(idata, group_name, None)
        if group is None:
            continue
        obj_vars = [v for v in group.data_vars if group[v].dtype == object]
        if obj_vars:
            setattr(idata, group_name, group.drop_vars(obj_vars))
            print(f"  [sanitize] dropped object-dtype vars from {group_name}: {obj_vars}")


def _log_marginal_likelihood(idata) -> float:
    """Extract the SMC log marginal likelihood (log Z) from an InferenceData.

    PyMC stores ``log_marginal_likelihood`` as an object array of lists, one
    list per chain × draw cell.  Each list holds the cumulative log-weight
    after each SMC stage; the final entry is the converged log Z for that
    chain.  This function extracts the last entry from every cell and returns
    the mean across chains.
    """
    lml = idata.sample_stats["log_marginal_likelihood"]
    raw = np.array(lml)  # (chains, draws), dtype=object or float

    # PyMC SMC stores log_marginal_likelihood in two different layouts depending
    # on whether chains ran in parallel (multiple cores) or serially:
    #
    #   Scalar layout  (chains, draws) of floats/NaN
    #     → each row is one chain; NaN-padded; last element is converged log Z.
    #     → occurs when chains > 1 and parallel execution is used.
    #
    #   List layout  (chains, draws) of Python lists
    #     → each cell holds the full per-stage history for one chain run.
    #     → occurs when chains = 1 or serial execution.
    #     → last element of each list is the converged log Z.
    #
    # In both cases we want the final converged value per chain.
    first = raw.flat[0]
    if isinstance(first, (list, np.ndarray)):
        # List layout: each cell is a stage-history list; take last element.
        chain_lmls = [float(cell[-1]) for cell in raw.flatten()]
    elif raw.ndim == 1:
        # Already sanitized (1-D float array, one value per chain).
        chain_lmls = [float(v) for v in np.array(raw, dtype=float)]
    else:
        # Scalar layout: rows are chains, columns are stages; take last column.
        chain_lmls = [float(v) for v in np.array(raw, dtype=float)[:, -1]]

    return float(np.nanmean(chain_lmls))


def compute_log_bf10(idata_m0, idata_m1) -> float:
    """log₁₀ BF₁₀ = (log Z_M1 − log Z_M0) / ln(10)."""
    return (_log_marginal_likelihood(idata_m1) - _log_marginal_likelihood(idata_m0)) / np.log(10)


def _posterior_summary(idata, var_name: str, hdi_prob: float = 0.95) -> tuple[float, float, float]:
    """Return (posterior mean, HDI lower, HDI upper) for a scalar posterior variable."""
    import arviz as az
    samples = np.array(idata.posterior[var_name]).flatten()
    mean    = float(samples.mean())
    hdi     = az.hdi(samples, hdi_prob=hdi_prob)
    return mean, float(hdi[0]), float(hdi[1])


def _compute_r2_bayesian(
    model: bmb.Model,
    idata,
    df_cond: pd.DataFrame,
    outcome_col: str,
) -> tuple[tuple, tuple]:
    """Compute Bayesian R² (Gelman et al. 2019) for conditional and marginal fits.

    For each posterior draw s:
        R²_s = Var(ŷ_s) / (Var(ŷ_s) + σ²_s)

    Conditional: ŷ includes fixed + random effects.
    Marginal:    ŷ includes fixed effects only (random effects zeroed out).

    Parameters
    ----------
    model : bambi.Model
        Compiled bambi model (M0 or M1).  Must have been built with
        ``model.build()`` prior to calling this function.
    idata : arviz.InferenceData
        Posterior samples returned by ``pm.sample_smc()`` for *model*.
        Must contain the ``posterior`` group with a ``"sigma"`` variable.
    df_cond : DataFrame
        Condition-specific dataframe used to fit *model*.  Passed to
        ``model.predict()`` as the new-data argument so that predictions
        are made on the same observations as the model was trained on.
    outcome_col : str
        Name of the outcome column in *df_cond* (e.g. ``"log_amplitude"``).
        Used only to determine the shape of the prediction array.

    Returns
    -------
    (cond_median, cond_lo, cond_hi), (marg_median, marg_lo, marg_hi)
        Each element is a 3-tuple of floats: posterior median R², 2.5th
        percentile, and 97.5th percentile across all chains × draws.
    """
    import copy

    # ── Conditional predictions — fixed + random effects ─────────────────────
    # bambi 0.17+: kind="response_params" (replaces deprecated kind="mean")
    # Predictions are stored as 'mu' in the posterior.
    idata_c = copy.deepcopy(idata)
    model.predict(idata_c, kind="response_params", include_group_specific=True, inplace=True)
    y_cond = np.array(idata_c.posterior["mu"])  # (chains, draws, obs)

    # ── Marginal predictions — fixed effects only ────────────────────────────
    idata_m = copy.deepcopy(idata)
    model.predict(idata_m, kind="response_params", include_group_specific=False, inplace=True)
    y_marg = np.array(idata_m.posterior["mu"])

    # Compute R² for each chain × draw by comparing the variance of the predictions
    sigma_flat   = np.array(idata.posterior["sigma"]).flatten()
    n_chains, n_draws, _ = y_cond.shape
    r2_cond = np.empty(n_chains * n_draws)
    r2_marg = np.empty(n_chains * n_draws)
    for idx, (chain, draw) in enumerate(np.ndindex(n_chains, n_draws)):
        sigma2 = float(sigma_flat[idx]) ** 2
        var_cond  = float(np.var(y_cond[chain, draw], ddof=0))
        var_marg  = float(np.var(y_marg[chain, draw], ddof=0))
        r2_cond[idx] = var_cond / (var_cond + sigma2)
        r2_marg[idx] = var_marg / (var_marg + sigma2)

    def _ci(arr):
        return (float(np.median(arr)),
                float(np.percentile(arr, 2.5)),
                float(np.percentile(arr, 97.5)))

    return _ci(r2_cond), _ci(r2_marg)


def extract_results(
    m0: bmb.Model,
    m1: bmb.Model,
    idata_m0,
    idata_m1,
    df_cond: pd.DataFrame,
    outcome_col: str = "amplitude",
    hdi_prob: float = 0.95,
) -> dict:
    """Extract all reportable quantities from fitted M0 and M1.

    Returns a flat dict with entries for fixed effects, random-effect SDs,
    correlations (NaN if not modelled), scalars (n, BF), R², amplitude, phase.
    """
    post1 = idata_m1.posterior
    nan3  = (np.nan, np.nan, np.nan)

    # ── Fixed effects ─────────────────────────────────────────────────────
    # Dict keys are the parameter names (α, β); the posterior variables carry
    # the regressor names (alpha_sin, beta_cos).
    fe = {}
    for key, var in [
        ("Intercept", "Intercept"),
        ("alpha",     "alpha_sin"),
        ("beta",      "beta_cos"),
    ]:
        fe[key] = _posterior_summary(idata_m1, var, hdi_prob) if var in post1 else nan3

    fe_m0_intercept = _posterior_summary(idata_m0, "Intercept", hdi_prob)

    # ── Random-effect SDs ─────────────────────────────────────────────────
    re_sd = {}
    for pmc_var, key in [
        ("1|participant_id_sigma",         "intercept"),
        ("alpha_sin|participant_id_sigma", "alpha"),
        ("beta_cos|participant_id_sigma",  "beta"),
    ]:
        re_sd[key] = _posterior_summary(idata_m1, pmc_var, hdi_prob) if pmc_var in post1 else nan3

    re_sd_m0_intercept = _posterior_summary(idata_m0, "1|participant_id_sigma", hdi_prob)

    # ── Random-effect correlations ────────────────────────────────────────
    # Bambi 0.17 with the (1 + alpha_sin + beta_cos | group) formula creates independent
    # random effects (separate sigma per slope); no correlation of the random effects
    # (LKJ correlation parameters) appear in the posterior, so these are reported as NaN.
    re_corr = {k: nan3 for k in ["intercept_alpha", "intercept_beta", "alpha_beta"]}

    # ── Scalars ───────────────────────────────────────────────────────────
    n_obs      = len(df_cond)
    n_clusters = df_cond[ANALYSIS_PARTICIPANT_COL].nunique()
    log_bf10   = compute_log_bf10(idata_m0, idata_m1)
    _, max_rhat_m0, max_rhat_m1 = compute_rhat_summary(idata_m0, idata_m1)

    # ── R² ────────────────────────────────────────────────────────────────
    print("    Computing R² for M1...")
    r2_cond_m1, r2_marg_m1 = _compute_r2_bayesian(m1, idata_m1, df_cond, outcome_col)
    print("    Computing R² for M0...")
    r2_cond_m0, r2_marg_m0 = _compute_r2_bayesian(m0, idata_m0, df_cond, outcome_col)

    # ── Amplitude and Phase from the fixed sin/cos slopes ─────────────────
    # y = μ + A·cos(ωt + φ), so A = sqrt(α² + β²) and φ = arctan2(-α, β),
    # in the same convention as the pupil and saliva cosinors.
    # α is the sine slope, β is the cosine slope.
    alpha_sin_samp = (np.array(post1["alpha_sin"]).flatten()
                      if "alpha_sin" in post1 else np.full(1, np.nan))
    beta_cos_samp  = (np.array(post1["beta_cos"]).flatten()
                      if "beta_cos" in post1 else np.full(1, np.nan))
    # Amplitude and phase are non-linear transforms of correlated α/β samples,
    # so CIs are derived by propagating the full joint posterior rather than combining
    # marginal CIs on α and β separately. Both quantities use fixed-effect samples
    # only (no random effects), so they describe population-level circadian modulation.
    # Mean is used for consistency with _posterior_summary (used for all other fixed effects).
    # Phase draws straddle +/-pi in several conditions, so the phase uses the circular
    # mean and SD, and its interval is taken on draws unwrapped around that mean.
    amp_samp      = np.sqrt(alpha_sin_samp**2 + beta_cos_samp**2)
    pha_samp      = np.arctan2(-alpha_sin_samp, beta_cos_samp)
    _q            = [(1 - hdi_prob) / 2 * 100, (1 + hdi_prob) / 2 * 100]
    amplitude_mean = float(np.mean(amp_samp))
    amplitude_sd   = float(np.std(amp_samp, ddof=1))
    phase_mean     = circ_mean(pha_samp)
    phase_sd       = circ_sd(pha_samp)
    amplitude_ci   = (float(np.percentile(amp_samp, _q[0])), float(np.percentile(amp_samp, _q[1])))
    phase_ci       = circ_interval(pha_samp, hdi_prob)

    # ── Back-transformed amplitude in mm (log-scale outcomes only) ─────────
    # For log-scale outcomes: A_mm = exp(intercept) * sinh(A_log)
    # This is the half-range of the back-transformed sinusoid (see utils.py).
    amplitude_mm: tuple | None = None

    # only if the outcome is on a log scale do we have a meaningful back-transformed amplitude in the original units; for other outcomes, this would be a meaningless transformation that depends on the arbitrary choice of zero point, so we leave it as None
    if outcome_col.startswith("log"):
        i_samp = np.array(post1["Intercept"]).flatten() if "Intercept" in post1 else None
        if i_samp is not None:
            amplitude_mm = backtransform_log_amplitude_circadian(
                i_samp, alpha_sin_samp, beta_cos_samp, hdi_prob)

    return dict(
        fe=fe, fe_m0_intercept=fe_m0_intercept,
        re_sd=re_sd, re_sd_m0_intercept=re_sd_m0_intercept,
        re_corr=re_corr,
        n_obs=n_obs, n_clusters=n_clusters, log_bf10=log_bf10,
        max_rhat_m0=max_rhat_m0, max_rhat_m1=max_rhat_m1,
        r2_cond_m1=r2_cond_m1, r2_marg_m1=r2_marg_m1,
        r2_cond_m0=r2_cond_m0, r2_marg_m0=r2_marg_m0,
        amplitude=amplitude_mean, amplitude_sd=amplitude_sd, amplitude_ci=amplitude_ci,
        phase_rad=phase_mean, phase_sd=phase_sd, phase_ci=phase_ci,
        amplitude_mm=amplitude_mm,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Table formatting and saving
# ═══════════════════════════════════════════════════════════════════════════

def format_bf_results_table(results: dict, condition: str) -> pd.DataFrame:
    """Build a DataFrame matching the reference table layout.

    Columns: Parameter | M1 Estimate | M1 2.5% | M1 97.5% | M0 Estimate | M0 2.5% | M0 97.5%

    Rows mirror the reference table:
        Fixed effects  : Intercept, β (sin), α (cos)
        Random effects : SD(Intercept), SD(β), SD(α), ρ(Intercept,β), ρ(Intercept,α), ρ(β,α)
        Scalars        : N_Observations, N_Clusters, log BF₁₀
        R²             : Conditional R², Marginal R²
        Derived        : Amplitude, Phase [rad]
    """
    nan = np.nan

    def make_row_with_ci(param, m1_ci, m0_ci=(nan, nan, nan)):
        est_m1, lo_m1, hi_m1 = m1_ci
        est_m0, lo_m0, hi_m0 = m0_ci
        return {
            "Parameter":   param,
            "M1 Estimate": round(est_m1, 3) if not np.isnan(est_m1) else nan,
            "M1 2.5%":     round(lo_m1,  3) if not np.isnan(lo_m1)  else nan,
            "M1 97.5%":    round(hi_m1,  3) if not np.isnan(hi_m1)  else nan,
            "M0 Estimate": round(est_m0, 3) if not np.isnan(est_m0) else nan,
            "M0 2.5%":     round(lo_m0,  3) if not np.isnan(lo_m0)  else nan,
            "M0 97.5%":    round(hi_m0,  3) if not np.isnan(hi_m0)  else nan,
        }

    def make_scalar_row(param, value):
        """Scalar row — single value, no CI columns."""
        return {
            "Parameter": param,
            "M1 Estimate": round(value, 3) if isinstance(value, float) else value,
            "M1 2.5%": nan, "M1 97.5%": nan,
            "M0 Estimate": nan, "M0 2.5%": nan, "M0 97.5%": nan,
        }

    fe   = results["fe"]
    re   = results["re_sd"]
    corr = results["re_corr"]

    rows = [
        # ── Fixed effects ─────────────────────────────────────────────────
        make_row_with_ci("Intercept",      fe["Intercept"], results["fe_m0_intercept"]),
        make_row_with_ci("α (sin)",        fe["alpha"]),
        make_row_with_ci("β (cos)",        fe["beta"]),
        # ── Random effects ────────────────────────────────────────────────
        make_row_with_ci("SD(Intercept)",  re["intercept"], results["re_sd_m0_intercept"]),
        make_row_with_ci("SD(α)",          re["alpha"]),
        make_row_with_ci("SD(β)",          re["beta"]),
        make_row_with_ci("ρ(Intercept, α)",corr["intercept_alpha"]),
        make_row_with_ci("ρ(Intercept, β)",corr["intercept_beta"]),
        make_row_with_ci("ρ(α, β)",        corr["alpha_beta"]),
        # ── Scalars ───────────────────────────────────────────────────────
        make_scalar_row("N_Observations", results["n_obs"]),
        make_scalar_row("N_Clusters",     results["n_clusters"]),
        make_scalar_row("log BF₁₀",       results["log_bf10"]),
        # ── R² ────────────────────────────────────────────────────────────
        make_row_with_ci("Conditional R²", results["r2_cond_m1"], results["r2_cond_m0"]),
        make_row_with_ci("Marginal R²",    results["r2_marg_m1"], results["r2_marg_m0"]),
        # ── Derived ───────────────────────────────────────────────────────
        make_scalar_row("Amplitude",      results["amplitude"]),
        make_scalar_row("Phase [rad]",    results["phase_rad"]),
    ]

    df = pd.DataFrame(rows)
    df.insert(0, "Condition", condition)
    return df


def results_to_row(results: dict, condition: str) -> dict:
    """Flatten an extract_results() dict into a single-row dict for CSV export.

    Naming convention:
        {m0|m1}_{parameter}_estimate / _lo / _hi   for posterior mean and 95% CI
        scalars (n_obs, n_clusters, log_bf10, amplitude, phase_rad) are kept as-is
    """
    def unpack(prefix, triple):
        est, lo, hi = triple
        return {f"{prefix}_estimate": est, f"{prefix}_lo": lo, f"{prefix}_hi": hi}

    row = {"condition": condition}

    fe = results["fe"]
    row.update(unpack("m1_intercept",       fe["Intercept"]))
    row.update(unpack("m1_alpha",           fe["alpha"]))
    row.update(unpack("m1_beta",            fe["beta"]))
    row.update(unpack("m0_intercept",       results["fe_m0_intercept"]))

    re = results["re_sd"]
    row.update(unpack("m1_re_sd_intercept", re["intercept"]))
    row.update(unpack("m1_re_sd_alpha",     re["alpha"]))
    row.update(unpack("m1_re_sd_beta",      re["beta"]))
    row.update(unpack("m0_re_sd_intercept", results["re_sd_m0_intercept"]))

    corr = results["re_corr"]
    row.update(unpack("re_corr_intercept_alpha", corr["intercept_alpha"]))
    row.update(unpack("re_corr_intercept_beta",  corr["intercept_beta"]))
    row.update(unpack("re_corr_alpha_beta",      corr["alpha_beta"]))

    row["n_obs"]        = results["n_obs"]
    row["n_clusters"]   = results["n_clusters"]
    row["log_bf10"]     = results["log_bf10"]
    row["max_rhat_m0"]  = results["max_rhat_m0"]
    row["max_rhat_m1"]  = results["max_rhat_m1"]

    row.update(unpack("m1_r2_cond", results["r2_cond_m1"]))
    row.update(unpack("m1_r2_marg", results["r2_marg_m1"]))
    row.update(unpack("m0_r2_cond", results["r2_cond_m0"]))
    row.update(unpack("m0_r2_marg", results["r2_marg_m0"]))

    row["amplitude"]        = results["amplitude"]
    row["amplitude_sd"]     = results["amplitude_sd"]
    row["amplitude_ci_lo"]  = results["amplitude_ci"][0]
    row["amplitude_ci_hi"]  = results["amplitude_ci"][1]
    row["phase_rad"]        = results["phase_rad"]
    row["phase_rad_sd"]     = results["phase_sd"]
    row["phase_rad_ci_lo"]  = results["phase_ci"][0]
    row["phase_rad_ci_hi"]  = results["phase_ci"][1]

    amp_mm = results.get("amplitude_mm")
    if amp_mm is not None:
        row["amplitude_mm_estimate"], row["amplitude_mm_lo"], row["amplitude_mm_hi"] = amp_mm
    else:
        row["amplitude_mm_estimate"] = row["amplitude_mm_lo"] = row["amplitude_mm_hi"] = np.nan

    return row


def _postprocess_tex_vars(tex: str) -> str:
    """Normalise raw output from format_latex_variables() for direct LaTeX inclusion.

    Two fixes applied:
    - Double-escaped backslashes (\\\\) produced by pandas string handling are
      collapsed to single backslashes so \\def commands render correctly.
    - \\acs{acronym} glossaries macros are expanded to plain \\Acronym commands,
      because the vars files are included in contexts that may not load glossaries.
    """
    tex = tex.replace("\\\\", "\\")
    return "\n".join(
        re.sub(
            r"\\acs\{([A-Za-z]+)\}",
            lambda m: "\\" + m.group(1)[0].upper() + m.group(1)[1:],
            line,
        )
        for line in tex.splitlines()
    )


def save_results(
    all_results: dict[str, dict],
    output_dir: Path,
    parameter_col: str = "amplitude",
) -> None:
    """Save all conditions as a single flat CSV and per-condition LaTeX tables.

    Also writes two combined-condition tables:
    - tab_parameter_estimates.tex  -- nested-row table of fixed/random effects + R2
    - tab_model_summary.tex        -- one row per condition: N, BF, amplitude, phase
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [results_to_row(res, cond) for cond, res in all_results.items()]
    combined = pd.DataFrame(rows)
    combined_path = output_dir / "circadian_pupil_all_conditions.csv"
    combined.to_csv(combined_path, index=False)
    print(f"  Saved result CSV: {combined_path.name}")

    # Output directory and macro prefix are namespaced by metric: save_results() runs
    # once per outcome metric, so a shared directory/prefix would let the last metric
    # silently overwrite the others and leave the manuscript citing the wrong model.
    # The per-metric subdirectory mirrors the tables/figures layout.
    tex_vars_path = (
        DEFAULT_OUTPUT_TEX_DIR / parameter_col / "vars_circadian_cosinor_parameters.tex"
    )
    tex_vars_path.parent.mkdir(parents=True, exist_ok=True)

    # Fixed-effect slopes are written at three decimals: at two decimals a credible
    # interval bound such as -0.004 renders as "-0.00", which reads as zero and hides
    # whether the interval excludes zero.
    slope_cols = [c for c in combined.columns if c.startswith(("m1_alpha", "m1_beta"))]
    other_cols = [c for c in combined.columns if c not in slope_cols and c != "condition"]
    prefix = f"circadian_cosinor_{parameter_col}"

    tex_vars_path.write_text(
        _postprocess_tex_vars(
            format_latex_variables(
                combined[["condition"] + other_cols], row_key_col="condition", prefix=prefix
            )
        )
        + "\n"
        + _postprocess_tex_vars(
            format_latex_variables(
                combined[["condition"] + slope_cols],
                row_key_col="condition",
                fmt=".3f",
                prefix=prefix,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"  Saved LaTeX variables: {tex_vars_path.name}")

    model_summary_cols = [
        "condition",
        "n_obs", "n_clusters", "log_bf10",
        "amplitude", "amplitude_sd", "amplitude_ci_lo", "amplitude_ci_hi",
        "phase_rad", "phase_rad_sd", "phase_rad_ci_lo", "phase_rad_ci_hi",
    ]
    summary_subset = combined[[c for c in model_summary_cols if c in combined.columns]]
    tex_vars_summary_path = (
        DEFAULT_OUTPUT_TEX_DIR / parameter_col / "vars_circadian_model_summary.tex"
    )
    tex_vars_summary_path.parent.mkdir(parents=True, exist_ok=True)
    tex_vars_summary_path.write_text(
        _postprocess_tex_vars(
            format_latex_variables(
                summary_subset,
                row_key_col="condition",
                prefix=f"circadian_model_summary_{parameter_col}",
            )
        ),
        encoding="utf-8",
    )
    print(f"  Saved LaTeX variables: {tex_vars_summary_path.name}")

    # for cond, res in all_results.items():
    #     latex_str = format_bf_results_latex(res, condition=cond)
    #     tex_path = output_dir / f"circadian_pupil_{cond}.tex"
    #     tex_path.write_text(latex_str, encoding="utf-8")
    #     print(f"  Saved result LaTeX: {tex_path.name}")

    param_est_tex = format_circadian_parameter_estimates_latex(
        all_results,
        parameter_col=parameter_col,
        label=f"tab:tab_circadian_parameter_estimates_{parameter_col}",
    )
    param_est_path = (
        output_dir / f"tab_parameter_estimates_{parameter_col}.tex"
    )
    param_est_path.write_text(param_est_tex, encoding="utf-8")
    print(f"  Saved combined LaTeX: {param_est_path.name}")

    model_sum_tex = format_circadian_model_summary_latex(all_results, parameter_col=parameter_col, label=f"tab:tab_circadian_model_summary_{parameter_col}")
    model_sum_path = output_dir / f"tab_model_summary_{parameter_col}.tex"
    model_sum_path.write_text(model_sum_tex, encoding="utf-8")
    print(f"  Saved combined LaTeX: {model_sum_path.name}")


def save_all_metrics_summary(
    metrics: list[str] | None = None,
    results_base_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Load per-metric result CSVs and write a combined model summary table.

    This aggregates the ``circadian_pupil_all_conditions.csv`` files produced
    by ``save_results()`` for each metric into one wide LaTeX table with one
    row per (metric, condition).

    Parameters
    ----------
    metrics :
        List of outcome parameter names to include. Defaults to all entries in
        ``PARAMETER_CHOICES``.
    results_base_dir :
        Parent directory containing one sub-directory per metric, each holding
        ``circadian_pupil_all_conditions.csv``. Defaults to
        ``results/tables/circadian_cosinor``.
    output_dir :
        Where to write ``tab_model_summary.tex``. Defaults to the
        same as ``results_base_dir``.
    """
    from civibeplr.analysis_io import PARAMETER_CHOICES

    if metrics is None:
        metrics = list(PARAMETER_CHOICES)
    if results_base_dir is None:
        results_base_dir = DEFAULT_OUTPUT_DIR
    if output_dir is None:
        output_dir = results_base_dir

    # Map column names in the flat CSV back to the keys expected by
    # format_circadian_model_summary_all_metrics_latex (which mirrors extract_results output).
    def _row_to_results(row: pd.Series) -> dict:
        amp_ci = (row.get("amplitude_ci_lo", np.nan), row.get("amplitude_ci_hi", np.nan))
        pha_ci = (row.get("phase_rad_ci_lo", np.nan), row.get("phase_rad_ci_hi", np.nan))
        return dict(
            n_obs=int(row["n_obs"]),
            n_clusters=int(row["n_clusters"]),
            log_bf10=float(row["log_bf10"]),
            amplitude=float(row["amplitude"]),
            amplitude_sd=float(row.get("amplitude_sd", np.nan)),
            amplitude_ci=amp_ci,
            phase_rad=float(row["phase_rad"]),
            phase_sd=float(row.get("phase_rad_sd", np.nan)),
            phase_ci=pha_ci,
        )

    all_metrics_results: dict[str, dict[str, dict]] = {}
    for metric in metrics:
        csv_path = results_base_dir / metric / "circadian_pupil_all_conditions.csv"
        if not csv_path.exists():
            print(f"  [skip] {metric}: {csv_path} not found")
            continue
        df = pd.read_csv(csv_path)
        cond_results = {}
        for _, row in df.iterrows():
            cond_results[row["condition"]] = _row_to_results(row)
        all_metrics_results[metric] = cond_results
        print(f"  Loaded {metric}: {len(cond_results)} conditions")

    if not all_metrics_results:
        print("  No metric results found; skipping combined summary table.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    tex = format_circadian_model_summary_all_metrics_latex(all_metrics_results)
    out_path = output_dir / "tab_model_summary.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"  Saved combined all-metrics summary: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(
    pupil_path: Path | None = None,
    saliva_path: Path | None = None,
    outcome_parameter: str = "amplitude",
    force: bool = False,
    dry_run: bool = False,
    smc_draws: int = DEFAULT_SMC_DRAWS,
    random_seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Load, prepare, build models, and fit or load InferenceData per condition.

    Parameters
    ----------
    force : bool
        If True, always re-run SMC sampling even when saved InferenceData
        already exists on disk.  If False (default), load existing .nc files
        and only fit conditions whose files are missing.
    dry_run : bool
        If True, stop after building and compiling models without fitting or
        loading any InferenceData.  Useful for checking model specification.
    smc_draws : int
        Number of SMC draws per model (default DEFAULT_SMC_DRAWS).
    random_seed : int
        Random seed for reproducibility.
    output_dir : Path | None
        Directory for saving result tables.  Defaults to
        results/tables/circadian_pupil/.

    Returns
    -------
    analysis_df : DataFrame
        Full prepared dataset with hours_since_start, log_amplitude, alpha_sin, beta_cos.
    result : dict
        If dry_run=False: {condition: result_dict}.
        If dry_run=True: {condition: (m0, m1)} compiled models.
    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / outcome_parameter
    idata_dir = DEFAULT_IDATA_DIR / outcome_parameter

    # ── Load ──────────────────────────────────────────────────────────────
    print("Loading data...")
    pupil_df  = load_pupil_results(pupil_path)
    saliva_df = load_saliva_results(saliva_path)
    print(f"  Pupil:  {len(pupil_df)} rows  |  "
          f"{pupil_df[PUPIL_PARTICIPANT_COL].nunique()} participants  |  "
          f"{pupil_df[PUPIL_CONDITION_COL].nunique()} conditions")
    print(f"  Saliva: {len(saliva_df)} participants  |  "
          f"τ range {saliva_df['tau_hours'].min():.2f}–{saliva_df['tau_hours'].max():.2f} h")

    # ── Merge and prepare ─────────────────────────────────────────────────
    merged_df = merge_pupil_saliva(pupil_df, saliva_df)
    analysis_df = build_analysis_df(merged_df)

    # ── Inspect outcome distribution before modelling ──────────────────────
    inspect_parameter_distribution(analysis_df, parameter_col=outcome_parameter)

    print(f"\nAnalysis dataframe: {len(analysis_df)} rows")
    print(analysis_df[
        [ANALYSIS_PARTICIPANT_COL, PUPIL_CONDITION_COL, PUPIL_BLOCK_COL,
         "hours_since_start", "amplitude", "log_amplitude", "alpha_sin", "beta_cos",
         "saliva_tau_hours"]
    ].head(10).to_string(index=False))

    # ── Build models ──────────────────────────────────────────────────────
    conditions = sorted(analysis_df[PUPIL_CONDITION_COL].dropna().unique())
    models: dict[str, tuple[bmb.Model, bmb.Model]] = {}
    condition_summary_rows: list[dict[str, int | str]] = []

    print(f"\nBuilding models for {len(conditions)} conditions...")
    for cond in conditions:
        df_cond = (
            analysis_df[analysis_df[PUPIL_CONDITION_COL] == cond]
            .dropna(subset=[outcome_parameter])
            .copy()
        )

        # Report sample sizes for this condition (after dropping missing outcome values)
        print(f"  [{cond}]  n(Observations)={len(df_cond)}  "
              f"n(Participants)={df_cond[ANALYSIS_PARTICIPANT_COL].nunique()}")
        condition_summary_rows.append(
            {
                "condition": cond,
                "n_observations": int(len(df_cond)),
                "n_participants": int(df_cond[ANALYSIS_PARTICIPANT_COL].nunique()),
            }
        )

        m0, m1 = build_models(df_cond, outcome_col=outcome_parameter)
        models[cond] = (m0, m1)

    # ── Per-condition sample-size LaTeX table ───────────────────────────
    output_dir.mkdir(parents=False, exist_ok=True)
    # Namespace the label by metric for the same reason as the macro prefixes: this
    # runs once per outcome, and a shared label makes LaTeX warn about duplicates and
    # resolve every \ref to whichever metric's table was typeset last.
    latex_path = output_dir / "tab_condition_sample_sizes.tex"
    latex_path.write_text(
        format_condition_sample_sizes_latex(
            condition_summary_rows,
            label=f"tab:tab_condition_sample_sizes_{outcome_parameter}",
        )
    )
    print(f"Saved LaTeX table to: {latex_path}")

    # ── Sampling + extraction ─────────────────────────────────────────────
    if dry_run:
        print("\nModels compiled (dry run).  Pass --force to force (re-)evaluation of models or omit --dry-run to fit or load.")
        return analysis_df, models

    all_results: dict[str, dict] = {}

    print(f"\nProcessing {len(models)} conditions (force={force})...")

    for cond, (m0, m1) in models.items():
        df_cond = (
            analysis_df[analysis_df[PUPIL_CONDITION_COL] == cond]
            .dropna(subset=[outcome_parameter, "alpha_sin", "beta_cos"])
            .copy()
        )

        idata_m0, idata_m1 = None, None
        if not force:
            idata_m0, idata_m1 = load_idata(cond, idata_dir=idata_dir)

        if idata_m0 is None:
            print(f"  [{cond}] Fitting model (SMC, {smc_draws} draws)...")
            idata_m0, idata_m1 = fit_condition(m0, m1, smc_draws=smc_draws, random_seed=random_seed)
            save_idata(m1, idata_m0, idata_m1, cond, outcome_col=outcome_parameter, idata_dir=idata_dir)

        print(f"\n── Condition: {cond.upper()} ──────────────────────────────")
        print("    Extracting results...")
        res = extract_results(m0, m1, idata_m0, idata_m1, df_cond, outcome_parameter)
        all_results[cond] = res

        print(format_bf_results_table(res, condition=cond).to_string(index=False))

    print(f"\nSaving results to {output_dir}...")
    save_results(all_results, output_dir, parameter_col=outcome_parameter)

    available_metrics = [
        p for p in PARAMETER_CHOICES
        if (DEFAULT_OUTPUT_DIR / p / "circadian_pupil_all_conditions.csv").exists()
    ]
    if len(available_metrics) > 1:
        print("\nUpdating combined all-metrics summary table...")
        save_all_metrics_summary()

    return analysis_df, all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Circadian pupil amplitude analysis.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run SMC sampling even when saved InferenceData already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and compile models only, without fitting or loading InferenceData.",
    )
    parser.add_argument("--smc-draws", type=int, default=DEFAULT_SMC_DRAWS)
    parser.add_argument("--seed", type=int, default=1072)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--outcome",
        nargs="+",
        choices=PARAMETER_CHOICES,
        default=PARAMETER_CHOICES,
        help=f"One or more outcome variables to analyse (default: all). Choices: {', '.join(PARAMETER_CHOICES)}.",
    )
    args = parser.parse_args()

    for outcome in args.outcome:
        if len(args.outcome) > 1:
            print(f"\n{'='*60}\nOutcome: {outcome}\n{'='*60}")
        main(
            force=args.force,
            dry_run=args.dry_run,
            smc_draws=args.smc_draws,
            random_seed=args.seed,
            output_dir=args.output_dir,
            outcome_parameter=outcome,
        )
