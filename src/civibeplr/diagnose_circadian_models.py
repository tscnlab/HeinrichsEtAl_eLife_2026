"""diagnose_circadian_models.py

Stage-2 model diagnostics for the circadian pupil parameter analysis.

Loads the InferenceData objects saved by ``analyse_circadian_pupil_parameters.py``
and runs eight diagnostic checks:

1. SMC log Z variance across chains
   Numerical + dot-plot per condition.  A high SD relative to the BF difference
   signals that the reported BF10 is numerically imprecise.

2. Prior predictive check
   Draws from the prior predictive distribution and overlays them against the
   observed data to verify that priors are weakly informative (not too tight
   or too diffuse).

3. Posterior predictive check (PPC)
   100 simulated datasets drawn from Normal(mu, sigma) are overlaid as KDE
   curves against the observed distribution.  A scatter of posterior-predicted
   mean vs. observed values is also shown.

4. Bayesian p-value (posterior predictive p-value)
   Computes p_B for two test statistics: mean(y) and SD(y).  Values near 0.5
   indicate good model fit; values near 0 or 1 indicate systematic discrepancy.

5. Posterior parameter summary
   Table of posterior mean, SD, and 95 % HDI for all fixed-effect parameters
   of M1.  Saved as CSV alongside the summary table.

6. Random effects shrinkage plot
   Participant-level posterior mean random intercepts vs. the unpooled
   (no-pooling) per-participant mean.  Shrinkage toward zero indicates
   appropriate partial pooling.

7. Residual analysis  (4-panel)
   Residuals = observed – posterior mean of conditional mu.
   Panels: histogram + normal overlay, Q-Q plot, residuals vs. fitted,
   residuals vs. time (block mean trend highlighted).

8. Within-participant ACF of residuals
   Residuals are indexed by protocol block within each participant and the
   ACF is computed at lags of 1-5 blocks (3.75 h each); pairs that span an
   excluded block are not used.  The mean ACF and 95 % bootstrap CI across
   participants are plotted with an approximate white-noise band for that mean.

Outputs
-------
results/figures/circadian_cosinor_diagnostics/<outcome_var>
    logz_variance_all_conditions.png
    diagnostics_{condition}.png  (combined 3x3 panel per condition)
results/tables/circadian_cosinor_diagnostics/<outcome_var>/
    diagnostics_summary.csv
    posterior_params_{condition}.csv

Usage
-----
python diagnose_circadian_models.py [--outcome log_amplitude] [--idata-dir PATH]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from civibeplr.vis_io import _in_jupyter, apply_font_style
from civibeplr.latex_table_creation import format_latex_variables, format_p_value

apply_font_style()

from civibeplr.analyse_circadian_pupil_parameters import (
    DEFAULT_IDATA_DIR,
)
from civibeplr.analysis_io import (
    ANALYSIS_PARTICIPANT_COL,
    PUPIL_CONDITION_COL,
    PUPIL_BLOCK_COL,
    BLOCK_INTERVAL_HOURS,
    get_parameter_label,
    lowercase_first_word,
)

# ── Default paths ─────────────────────────────────────────────────────────────

_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DIAG_FIGURES_DIR = (
    _PROJECT_DIR / "results" / "figures" / "circadian_cosinor_diagnostics"
)
DEFAULT_DIAG_TABLES_DIR = _PROJECT_DIR / "results" / "tables" / "circadian_cosinor_diagnostics"
DEFAULT_DIAG_TEX_DIR = (
    _PROJECT_DIR / "results" / "tex" / "circadian_cosinor_diagnostics"
)


# ═══════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_idata(condition: str, idata_dir: Path | None = None) -> tuple:
    """Load M0 and M1 InferenceData for *condition* from NetCDF files."""
    import arviz as az

    if idata_dir is None:
        idata_dir = DEFAULT_IDATA_DIR
    cond_safe = condition.replace(" ", "_").lower()
    path_m0 = idata_dir / f"{cond_safe}_M0.nc"
    path_m1 = idata_dir / f"{cond_safe}_M1.nc"
    if not path_m0.exists() or not path_m1.exists():
        raise FileNotFoundError(
            f"InferenceData not found for condition '{condition}'.\n"
            f"Expected:\n  {path_m0}\n  {path_m1}\n"
            "Run analyse_circadian_pupil_parameters.py --fit first."
        )
    return az.from_netcdf(str(path_m0)), az.from_netcdf(str(path_m1))


def load_condition_data(condition: str, idata_dir: Path | None = None) -> pd.DataFrame:
    """Load the training DataFrame saved alongside the InferenceData."""
    if idata_dir is None:
        idata_dir = DEFAULT_IDATA_DIR
    cond_safe = condition.replace(" ", "_").lower()
    path = idata_dir / f"{cond_safe}_data.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Training data CSV not found: {path}\n"
            "Run analyse_circadian_pupil_parameters.py --fit first."
        )
    return pd.read_csv(path).reset_index(drop=True)


def load_rhat_summary(
    condition: str,
    idata_dir: Path | None = None,
    idata_m0=None,
    idata_m1=None,
) -> tuple[float, float]:
    """Return (max_rhat_m0, max_rhat_m1) for *condition*.

    Reads from the pre-computed ``_rhat.csv`` if it exists.  Falls back to
    computing via ``compute_rhat_summary`` when the file is absent (e.g. for
    models fitted before this feature was added), which requires *idata_m0*
    and *idata_m1* to be passed in.
    """
    if idata_dir is None:
        idata_dir = DEFAULT_IDATA_DIR
    cond_safe = condition.replace(" ", "_").lower()
    rhat_path = idata_dir / f"{cond_safe}_rhat.csv"

    if rhat_path.exists():
        df = pd.read_csv(rhat_path)
        max_m0 = float(df.loc[df["model"] == "M0", "rhat"].max())
        max_m1 = float(df.loc[df["model"] == "M1", "rhat"].max())
        return max_m0, max_m1

    if idata_m0 is None or idata_m1 is None:
        return np.nan, np.nan

    from civibeplr.analyse_circadian_pupil_parameters import compute_rhat_summary
    _, max_m0, max_m1 = compute_rhat_summary(idata_m0, idata_m1)
    return max_m0, max_m1


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _per_chain_logz(idata) -> list[float]:
    """Extract the converged log Z value for each SMC chain."""
    lml = idata.sample_stats["log_marginal_likelihood"]
    raw = np.array(lml)
    first = raw.flat[0]
    if isinstance(first, (list, np.ndarray)):
        # List layout: each cell contains the per-stage history; last entry = log Z
        return [float(cell[-1]) for cell in raw.flatten()]
    elif raw.ndim == 1:
        # Already sanitized (saved via _sanitize_idata_for_netcdf): 1-D float array
        return [float(v) for v in np.array(raw, dtype=float)]
    else:
        # Scalar layout: rows = chains, last column = converged log Z
        return [float(v) for v in np.array(raw, dtype=float)[:, -1]]


def _get_residuals(idata, df: pd.DataFrame, outcome_col: str) -> np.ndarray:
    """Return observed – posterior mean of conditional mu."""
    mu = np.array(idata.posterior["mu"])  # (chains, draws, n_obs)
    return df[outcome_col].values - mu.mean(axis=(0, 1))


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic 1 — SMC log Z variance
# ═══════════════════════════════════════════════════════════════════════════

def check_logz_variance(
    idata_m0,
    idata_m1,
    condition: str,
    ax: "plt.Axes | None" = None,
) -> dict:
    """Assess BF reliability via per-chain log Z spread.

    Plots each chain's log Z for M0 and M1 with mean ± SD error bars.
    Returns a dict with summary statistics including the ratio of pooled SD
    to the BF difference in nats (``bf_SD_over_difference``).  Values >= 0.1
    mark a numerically imprecise Bayes factor.  This measures scatter between
    SMC populations only; it does not detect particle-count bias in log Z.
    """
    lz_m0 = _per_chain_logz(idata_m0)
    lz_m1 = _per_chain_logz(idata_m1)

    mean_m0, sd_m0 = float(np.nanmean(lz_m0)), float(np.nanstd(lz_m0))
    mean_m1, sd_m1 = float(np.nanmean(lz_m1)), float(np.nanstd(lz_m1))

    log_bf10       = (mean_m1 - mean_m0) / np.log(10)
    bf_diff_nats   = mean_m1 - mean_m0
    pooled_sd      = float(np.sqrt(sd_m0 ** 2 + sd_m1 ** 2))
    reliability    = pooled_sd / abs(bf_diff_nats) if bf_diff_nats != 0 else np.inf

    if ax is not None:
        ax.scatter(np.zeros(len(lz_m0)), lz_m0, s=60, color="steelblue",
                   label="M0", zorder=3)
        ax.scatter(np.ones(len(lz_m1)),  lz_m1, s=60, color="darkorange",
                   label="M1", zorder=3)
        ax.errorbar([0], [mean_m0], yerr=[sd_m0], fmt="none",
                    color="steelblue", capsize=8, lw=2)
        ax.errorbar([1], [mean_m1], yerr=[sd_m1], fmt="none",
                    color="darkorange", capsize=8, lw=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["M0 (null)", "M1 (cosinor)"])
        ax.set_ylabel("log Z (per chain)")
        ax.set_title(
            f"{condition}\nlog BF10 = {log_bf10:.2f}   "
            f"SD/|ΔlogZ| = {reliability:.3f}",
            #fontsize=10,
        )
        ax.legend()  # fontsize=9

    return dict(
        condition=condition,
        logZ_M0_mean=round(mean_m0, 3),
        logZ_M0_SD=round(sd_m0, 3),
        logZ_M1_mean=round(mean_m1, 3),
        logZ_M1_SD=round(sd_m1, 3),
        log_bf10=round(log_bf10, 3),
        bf_SD_over_difference=round(reliability, 4),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic 2 — Prior predictive check
# ═══════════════════════════════════════════════════════════════════════════

def plot_prior_predictive(
    idata_m1,
    df: pd.DataFrame,
    condition: str,
    outcome_col: str = "amplitude",
    n_rep: int = 200,
    figures_dir: Path | None = None,
    rng_seed: int = 42,
    ax: "plt.Axes | None" = None,
) -> None:
    """Prior predictive check: are the priors weakly informative?

    Draws *n_rep* datasets from the prior predictive distribution stored in
    ``idata_m1.prior_predictive`` (if available) or approximated from the
    prior on sigma and a grand-mean draw.  Overlays their KDE against the
    observed distribution.

    A good prior should span the observed range but not be orders of magnitude
    wider — the prior KDE envelope should comfortably contain the orange
    observed curve without being orders of magnitude wider.

    If *ax* is provided the plot is drawn into that axes and no file is saved.
    """
    from scipy.stats import gaussian_kde

    save_fig = ax is None
    if figures_dir is None:
        figures_dir = DEFAULT_DIAG_FIGURES_DIR
    if save_fig:
        figures_dir.mkdir(parents=True, exist_ok=True)

    y_obs = df[outcome_col].values
    cond_safe = condition.replace(" ", "_").lower()

    has_prior_pred = (
        hasattr(idata_m1, "prior_predictive")
        and idata_m1.prior_predictive is not None
        and outcome_col in idata_m1.prior_predictive
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = None
    xgrid = np.linspace(y_obs.min() - 2.0, y_obs.max() + 2.0, 400)

    if has_prior_pred:
        y_prior = np.array(idata_m1.prior_predictive[outcome_col])
        y_prior_flat = y_prior.reshape(-1, y_prior.shape[-1])
        rng = np.random.default_rng(rng_seed)
        idx = rng.choice(len(y_prior_flat), size=min(n_rep, len(y_prior_flat)), replace=False)
        for j, i in enumerate(idx):
            try:
                ax.plot(xgrid, gaussian_kde(y_prior_flat[i])(xgrid),
                        color="steelblue", alpha=max(0.01, 2.0 / n_rep), lw=1,
                        label="Prior predictive draws" if j == 0 else None)
            except Exception:
                pass

    else:
        # Approximate from prior parameters stored in idata_m1.prior
        has_prior = hasattr(idata_m1, "prior") and idata_m1.prior is not None
        prior = idata_m1.prior if has_prior else None
        intercept_key = next(
            (k for k in prior.data_vars if "intercept" in k.lower()), None
        ) if has_prior else None
        sigma_key = next(
            (k for k in prior.data_vars if k.lower() in ("sigma", "y_sigma")), None
        ) if has_prior else None
        rng = np.random.default_rng(rng_seed)
        n_drawn = 0
        if intercept_key and sigma_key:
            intercepts = np.array(prior[intercept_key]).flatten()
            sigmas = np.array(prior[sigma_key]).flatten()
            n_avail = min(len(intercepts), len(sigmas))
            idx = rng.choice(n_avail, size=min(n_rep, n_avail), replace=False)
            for j, i in enumerate(idx):
                y_draw = rng.normal(intercepts[i], sigmas[i], size=len(y_obs))
                try:  
                    # source_label = "Prior draws (approximated)"
                    ax.plot(xgrid, gaussian_kde(y_draw)(xgrid),
                            color="steelblue", alpha=max(0.01, 2.0 / n_rep), lw=1,
                            label="Prior predictive draws" if j == 0 else None)
                    n_drawn += 1
                except Exception:
                    pass
        if n_drawn == 0:
            ax.text(0.5, 0.5, "Prior predictive data not available.\nRe-run with prior_predictive=True.",
                    transform=ax.transAxes, ha="center", va="center")#, fontsize=10)

    ax.plot(xgrid, gaussian_kde(y_obs)(xgrid),
            color="darkorange", lw=2.5, label="Observed")
    ax.set_xlabel(get_parameter_label(outcome_col))
    ax.set_ylabel("Density")
    ax.set_title(
        f"Prior predictive check"
        #f"Blue = {source_label} | Orange = observed"
    )
    ax.legend()#fontsize=9)
    if save_fig:
        fig.tight_layout()
        path = figures_dir / f"prior_predictive_{cond_safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if not _in_jupyter():
            plt.close(fig)
        print(f"  Saved: {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic 3 — Posterior predictive check (PPC)
# ═══════════════════════════════════════════════════════════════════════════

def plot_ppc(
    idata_m1,
    df: pd.DataFrame,
    condition: str,
    outcome_col: str = "amplitude",
    n_rep: int = 100,
    ax_density: "plt.Axes | None" = None,
    ax_scatter: "plt.Axes | None" = None,
    rng_seed: int = 42,
) -> None:
    """Posterior predictive check via manual draws from Normal(mu, sigma).

    Left panel  — KDE of observed log-amplitude vs. n_rep simulated datasets
                  (blue = simulated, orange = observed).
    Right panel — Posterior predicted mean E[y_rep | θ] vs. observed scatter.
                  Points on the diagonal indicate good prediction.
    """
    from scipy.stats import gaussian_kde

    y_obs         = df[outcome_col].values
    mu_samples    = np.array(idata_m1.posterior["mu"])       # (chains, draws, n_obs)
    sigma_samples = np.array(idata_m1.posterior["sigma"])    # (chains, draws)

    n_chains, n_draws, n_obs = mu_samples.shape
    mu_flat    = mu_samples.reshape(-1, n_obs)
    sigma_flat = sigma_samples.reshape(-1)

    rng     = np.random.default_rng(rng_seed)
    idx_rep = rng.choice(len(mu_flat), size=n_rep, replace=False)
    y_rep   = rng.normal(mu_flat[idx_rep], sigma_flat[idx_rep, None])  # (n_rep, n_obs)
    y_rep_mean = mu_flat.mean(axis=0)                                   # E[mu | data]

    if ax_density is None or ax_scatter is None:
        _, (ax_density, ax_scatter) = plt.subplots(1, 2, figsize=(12, 5))

    # ── KDE density overlay ───────────────────────────────────────────────
    xgrid = np.linspace(y_obs.min() - 0.5, y_obs.max() + 0.5, 300)
    for i in range(n_rep):
        try:
            ax_density.plot(xgrid, gaussian_kde(y_rep[i])(xgrid),
                            color="steelblue", alpha=max(0.02, 3.0 / n_rep), lw=1,
                            label="Posterior predictive draws" if i == 0 else None)
        except Exception:
            pass
    ax_density.plot(xgrid, gaussian_kde(y_obs)(xgrid),
                    color="darkorange", lw=2.5, label="Observed")
    ax_density.set_xlabel(get_parameter_label(outcome_col))
    ax_density.set_ylabel("Density")
    ax_density.set_title(f"PPC density ({condition.upper()})")
    ax_density.legend()#fontsize=9)

    # ── E[y_rep | θ] vs. observed ─────────────────────────────────────────
    ax_scatter.scatter(y_obs, y_rep_mean, s=25, alpha=0.6, color="steelblue")
    lims = [
        min(y_obs.min(), y_rep_mean.min()) - 0.1,
        max(y_obs.max(), y_rep_mean.max()) + 0.1,
    ]
    ax_scatter.plot(lims, lims, "r--", lw=1, alpha=0.7)
    ax_scatter.set_xlabel("Observed")
    ax_scatter.set_ylabel("E[y_rep | θ]")
    ax_scatter.set_title(f"PPC predicted vs. observed ({condition.upper()})")


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic 4 — Bayesian p-value (posterior predictive p-value)
# ═══════════════════════════════════════════════════════════════════════════

def compute_bayesian_pvalue(
    idata,
    df: pd.DataFrame,
    condition: str,
    outcome_col: str = "amplitude",
    rng_seed: int = 42,
    suffix: str = "",
) -> dict:
    """Compute Bayesian posterior predictive p-values for a model.

    For two test statistics T -- mean(y) and SD(y) -- the Bayesian p-value is:
        p_B = P(T(y_rep) >= T(y_obs) | y_obs)

    Values near 0.5 indicate that the model generates data similar to the
    observed in that statistic.  Values near 0 or 1 indicate systematic
    discrepancy (model under- or over-estimates that aspect of the data).

    *suffix* is appended to output keys (e.g. ``"_m0"`` -> ``pB_mean_m0``).

    Returns
    -------
    dict : pB_mean[suffix], pB_sd[suffix]
    """
    y_obs = df[outcome_col].values
    mu_samples = np.array(idata.posterior["mu"])       # (chains, draws, n_obs)
    sigma_samples = np.array(idata.posterior["sigma"]) # (chains, draws)

    n_chains, n_draws, n_obs = mu_samples.shape
    mu_flat = mu_samples.reshape(-1, n_obs)
    sigma_flat = sigma_samples.reshape(-1)

    rng = np.random.default_rng(rng_seed)
    n_rep = len(mu_flat)
    y_rep = rng.normal(mu_flat, sigma_flat[:, None])  # (n_rep, n_obs)

    T_mean_obs = float(np.mean(y_obs))
    T_sd_obs   = float(np.std(y_obs))

    T_mean_rep = np.mean(y_rep, axis=1)
    T_sd_rep   = np.std(y_rep, axis=1)

    pB_mean = float(np.mean(T_mean_rep >= T_mean_obs))
    pB_sd   = float(np.mean(T_sd_rep   >= T_sd_obs))

    return dict(
        condition=condition,
        **{
            f"pB_mean{suffix}": round(pB_mean, 4),
            f"pB_sd{suffix}":   round(pB_sd,   4),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic 5 — Posterior parameter summary
# ═══════════════════════════════════════════════════════════════════════════

def summarise_posterior(
    idata_m1,
    condition: str,
    outcome_col: str = "amplitude",
    hdi_prob: float = 0.95,
    tables_dir: Path | None = None,
) -> pd.DataFrame:
    """Summarise M1 fixed-effect posteriors: mean, SD, 95 % HDI.

    For SMC there is no R-hat or ESS in the MCMC sense; ESS is approximated
    from the effective sample size of the SMC particle weights when available
    (stored in ``sample_stats.ess``), otherwise omitted.

    Saves a CSV per condition and returns the DataFrame.
    """
    import arviz as az

    if tables_dir is None:
        tables_dir = DEFAULT_DIAG_TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)

    fixed_vars = [v for v in idata_m1.posterior.data_vars
                  if not v.startswith(f"1|") and "|" not in v
                  and not v.endswith("_sigma") and v != "mu"]

    rows = []
    for var in fixed_vars:
        samples = np.array(idata_m1.posterior[var]).flatten()
        hdi = az.hdi(samples, hdi_prob=hdi_prob)
        rows.append(dict(
            parameter=var,
            mean=round(float(np.mean(samples)), 4),
            sd=round(float(np.std(samples)), 4),
            hdi_lower=round(float(hdi[0]), 4),
            hdi_upper=round(float(hdi[1]), 4),
            hdi_prob=hdi_prob,
        ))

    df_params = pd.DataFrame(rows)
    cond_safe = condition.replace(" ", "_").lower()
    path = tables_dir / f"posterior_params_{cond_safe}.csv"
    df_params.to_csv(path, index=False)
    print(f"  Saved: {path.name}")
    return df_params


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic 6 — Random effects shrinkage plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_shrinkage(
    idata_m1,
    df: pd.DataFrame,
    condition: str,
    outcome_col: str = "amplitude",
    figures_dir: Path | None = None,
    ax: "plt.Axes | None" = None,
) -> None:
    """Random-intercept shrinkage plot: partial-pooling vs. no-pooling.

    Each point is one participant.  X-axis = no-pooling estimate (raw
    per-participant mean of the outcome).  Y-axis = posterior mean of the
    participant-level random intercept (partial-pooling estimate).

    Shrinkage toward the grand mean (diagonal line) is expected and desirable;
    extreme shrinkage of specific participants may indicate influential
    observations or prior misspecification.

    If *ax* is provided the plot is drawn into that axes and no file is saved.
    """
    save_fig = ax is None
    if figures_dir is None:
        figures_dir = DEFAULT_DIAG_FIGURES_DIR
    if save_fig:
        figures_dir.mkdir(parents=True, exist_ok=True)

    re_key = next(
        (k for k in idata_m1.posterior.data_vars
         if "1|" in k or ("|" in k and "participant" in k.lower()
                           and "alpha_sin" not in k and "beta_cos" not in k)),
        None,
    )
    if re_key is None:
        print(f"  [Shrinkage] Random intercept variable not found - skipping.")
        return

    re_samples = np.array(idata_m1.posterior[re_key])  # (chains, draws, n_participants)
    re_mean = re_samples.mean(axis=(0, 1))              # (n_participants,)

    # Recover participant ordering from the InferenceData coordinate
    coord_key = next(
        (k for k in idata_m1.posterior[re_key].dims if k != "chain" and k != "draw"),
        None,
    )
    if coord_key is not None and coord_key in idata_m1.posterior.coords:
        pids = list(idata_m1.posterior.coords[coord_key].values)
    else:
        pids = list(range(len(re_mean)))

    # No-pooling: raw per-participant mean of outcome (centred by grand mean)
    grand_mean = float(df[outcome_col].mean())
    part_means = df.groupby(ANALYSIS_PARTICIPANT_COL)[outcome_col].mean()
    # pids from InferenceData coords may be integers (0-indexed) while
    # part_means is indexed by the actual participant IDs -- align by position.
    if set(pids).issubset(set(part_means.index)):
        no_pool = part_means.reindex(pids).values - grand_mean
    else:
        no_pool = part_means.values - grand_mean

    cond_safe = condition.replace(" ", "_").lower()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = None
    ax.scatter(no_pool, re_mean, s=50, color="steelblue", alpha=0.8, zorder=3)
    lims = [
        min(no_pool.min(), re_mean.min()) - 0.05,
        max(no_pool.max(), re_mean.max()) + 0.05,
    ]
    ax.plot(lims, lims, "r--", lw=1, alpha=0.6, label="No shrinkage")
    ax.axhline(0, color="grey", lw=0.8, linestyle=":")
    ax.axvline(0, color="grey", lw=0.8, linestyle=":")
    ax.set_xlabel("No-pooling estimate\n(raw participant mean - grand mean)")
    ax.set_ylabel("Partial-pooling estimate\n(posterior mean random intercept)")
    ax.set_title("Random-intercept shrinkage")
    ax.legend()#fontsize=9)
    if save_fig:
        fig.tight_layout()
        path = figures_dir / f"shrinkage_{cond_safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if not _in_jupyter():
            plt.close(fig)
        print(f"  Saved: {path.name}")


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic 7 — Residual analysis
# ═══════════════════════════════════════════════════════════════════════════

def plot_residuals(
    idata,
    df: pd.DataFrame,
    condition: str,
    outcome_col: str = "amplitude",
    figures_dir: Path | None = None,
    axes: "np.ndarray | None" = None,
    suffix: str = "",
) -> dict:
    """Four-panel residual diagnostic.

    Panels
    ------
    1. Histogram of residuals + fitted normal curve
    2. Q-Q plot (residuals vs. theoretical normal quantiles)
    3. Residuals vs. posterior mean fitted values  (homoscedasticity)
    4. Residuals vs. hours_since_start with block-mean trend line  (temporal structure)

    If *axes* is provided (a flat array of 4 Axes), plots are drawn into those
    axes and no file is saved.

    *suffix* is appended to output keys (e.g. ``"_m0"`` -> ``residual_mean_m0``).

    Returns
    -------
    dict : residual_mean[suffix], residual_SD[suffix], shapiro_W[suffix], shapiro_p[suffix]
    """
    from scipy import stats

    save_fig = axes is None
    if figures_dir is None:
        figures_dir = DEFAULT_DIAG_FIGURES_DIR
    if save_fig:
        figures_dir.mkdir(parents=True, exist_ok=True)

    residuals = _get_residuals(idata, df, outcome_col)
    mu_mean   = np.array(idata.posterior["mu"]).mean(axis=(0, 1))
    cond_safe = condition.replace(" ", "_").lower()

    time_col  = "hours_since_start" if "hours_since_start" in df.columns else PUPIL_BLOCK_COL

    if axes is None:
        fig, axes_2d = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Residual diagnostics: M1, {condition}")#, fontsize=13)
        axes = axes_2d.flat
    else:
        fig = None
        axes = np.asarray(axes).flat

    axes_list = list(axes)

    # ── 1. Histogram + normal overlay ────────────────────────────────────
    ax = axes_list[0]
    ax.hist(residuals, bins=20, density=True,
            color="steelblue", alpha=0.7, edgecolor="white")
    x_norm = np.linspace(residuals.min() - 0.2, residuals.max() + 0.2, 300)
    ax.plot(x_norm, stats.norm.pdf(x_norm, residuals.mean(), residuals.std()),
            "r-", lw=2, label="Normal fit")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of residuals")
    ax.legend()

    # ── 2. Q-Q plot ───────────────────────────────────────────────────────
    ax = axes_list[1]
    (osm, osr), (slope, intercept, _) = stats.probplot(residuals)
    ax.scatter(osm, osr, s=25, color="steelblue", alpha=0.7)
    qq_x = np.array([osm.min(), osm.max()])
    ax.plot(qq_x, slope * qq_x + intercept, "r-", lw=2)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title("Q-Q plot (residuals vs. Normal)")

    # ── 3. Residuals vs. fitted ───────────────────────────────────────────
    ax = axes_list[2]
    ax.scatter(mu_mean, residuals, s=25, color="steelblue", alpha=0.6)
    ax.axhline(0, color="r", lw=1.5, linestyle="--")
    ax.set_xlabel("Posterior mean ŷ (conditional)")
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs. fitted values")

    # ── 4. Residuals vs. time + block-mean trend ──────────────────────────
    ax = axes_list[3]
    ax.scatter(df[time_col].values, residuals, s=25, color="steelblue", alpha=0.6)
    ax.axhline(0, color="r", lw=1.5, linestyle="--")
    block_mean_resid = (
        df.assign(_resid=residuals)
        .groupby(PUPIL_BLOCK_COL)
        .agg(_resid_mean=("_resid", "mean"), _time_mean=(time_col, "mean"))
        .reset_index()
        .sort_values("_time_mean")
    )
    ax.plot(block_mean_resid["_time_mean"].values,
            block_mean_resid["_resid_mean"].values,
            "r-o", lw=1.5, markersize=5, label="Block mean")
    ax.set_xlabel(get_parameter_label(time_col))
    ax.set_ylabel("Residual")
    ax.set_title("Residuals vs. time")
    ax.legend()#fontsize=9)

    if save_fig:
        fig.tight_layout()
        path = figures_dir / f"residuals_{cond_safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if not _in_jupyter():
            plt.close(fig)
        print(f"  Saved: {path.name}")

    sw_stat, sw_p = stats.shapiro(residuals)
    return dict(
        condition=condition,
        **{
            f"residual_mean{suffix}": round(float(residuals.mean()), 4),
            f"residual_SD{suffix}":   round(float(residuals.std()),  4),
            f"shapiro_W{suffix}":     round(float(sw_stat),          4),
            # Unrounded: small p-values must survive to the table formatter,
            # which prints them as "<0.001" rather than as zero.
            f"shapiro_p{suffix}":     float(sw_p),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic 8 — Within-participant ACF of residuals
# ═══════════════════════════════════════════════════════════════════════════

def plot_acf_within_participant(
    idata,
    df: pd.DataFrame,
    condition: str,
    outcome_col: str = "amplitude",
    max_lag: int = 5,
    figures_dir: Path | None = None,
    ax: "plt.Axes | None" = None,
    suffix: str = "",
) -> dict:
    """Within-participant autocorrelation of residuals at fixed block lags.

    For each participant, residuals are indexed by protocol block and the ACF
    at lag *k* uses only pairs exactly *k* blocks (k x 3.75 h) apart, so a
    pair never spans an excluded block.  The standard (biased, 1/n) estimator
    is used: products are summed over the available pairs and divided by
    n x variance, n being the participant's number of retained blocks.  With
    7-11 blocks per participant the lag-1 ACF of uncorrelated residuals is
    expected near -1/n rather than 0.

    The mean ACF and 95 % bootstrap CI across participants are shown as a bar
    chart.  The dashed band is an approximate 95 % white-noise range for the
    participant mean: each participant ACF has variance ~1/n_i under white
    noise, so the mean of m participants has SD ~sqrt(sum 1/n_i) / m.

    If *ax* is provided the plot is drawn into that axes and no file is saved.

    *suffix* is appended to output keys (e.g. ``"_m0"`` -> ``lag1_mean_m0``).

    Returns
    -------
    dict : lag1_mean[suffix], lag1_SD[suffix]  (across participants)
    """
    save_fig = ax is None
    if figures_dir is None:
        figures_dir = DEFAULT_DIAG_FIGURES_DIR
    if save_fig:
        figures_dir.mkdir(parents=True, exist_ok=True)

    df_work = df.copy().reset_index(drop=True)
    df_work["_resid"] = _get_residuals(idata, df_work, outcome_col)
    if PUPIL_BLOCK_COL in df_work.columns:
        df_work["_block"] = df_work[PUPIL_BLOCK_COL].astype(int)
    else:
        df_work["_block"] = np.rint(
            df_work["hours_since_start"] / BLOCK_INTERVAL_HOURS
        ).astype(int)
    df_work = df_work.sort_values([ANALYSIS_PARTICIPANT_COL, "_block"])

    acf_rows = []
    n_obs = []
    for pid in df_work[ANALYSIS_PARTICIPANT_COL].unique():
        sub = df_work[df_work[ANALYSIS_PARTICIPANT_COL] == pid]
        n = len(sub)
        if n < 3:
            continue
        resid = sub["_resid"].values - sub["_resid"].values.mean()
        var = np.var(resid)
        if var < 1e-10:
            continue
        # Place residuals on the full block grid so that a lag of k always
        # means k blocks; excluded blocks stay NaN and drop out of the pairs.
        blocks = sub["_block"].values
        grid = np.full(blocks.max() - blocks.min() + 1, np.nan)
        grid[blocks - blocks.min()] = resid
        row = []
        for lag in range(1, max_lag + 1):
            if lag >= len(grid):
                row.append(np.nan)
                continue
            prod = grid[:-lag] * grid[lag:]
            if np.isnan(prod).all():
                row.append(np.nan)
                continue
            row.append(float(np.nansum(prod) / (n * var)))
        acf_rows.append(row)
        n_obs.append(n)

    if not acf_rows:
        print(f"  [ACF] Insufficient data for '{condition}'.")
        return dict(condition=condition, **{f"lag1_mean{suffix}": np.nan, f"lag1_SD{suffix}": np.nan})

    acf_arr = np.array(acf_rows)   # (n_participants, max_lag)
    lags     = np.arange(1, max_lag + 1)
    acf_mean = np.nanmean(acf_arr, axis=0)

    # 95 % bootstrap CI across participants
    rng = np.random.default_rng(42)
    boot = np.array([
        np.nanmean(
            acf_arr[rng.choice(len(acf_arr), len(acf_arr), replace=True)], axis=0
        )
        for _ in range(2000)
    ])
    ci_lo = np.percentile(boot, 2.5,  axis=0)
    ci_hi = np.percentile(boot, 97.5, axis=0)

    white_noise_band = 1.96 * np.sqrt(np.sum(1.0 / np.array(n_obs))) / len(n_obs)
    cond_safe = condition.replace(" ", "_").lower()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = None
    ax.bar(lags, acf_mean, color="steelblue", alpha=0.8, label="Mean ACF")
    ax.errorbar(
        lags, acf_mean,
        yerr=[acf_mean - ci_lo, ci_hi - acf_mean],
        fmt="none", color="black", capsize=5, lw=1.5,
        label="95 % bootstrap CI",
    )
    ax.axhline( white_noise_band, color="r", linestyle="--", lw=1.2, alpha=0.7,
                label="Approx. 95 % white-noise range")
    ax.axhline(-white_noise_band, color="r", linestyle="--", lw=1.2, alpha=0.7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Lag [blocks of 3.75 h]")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(
        f"Within-participant residual ACF\n"
        f"(n = {len(acf_rows)} participants)"
    )
    ax.set_xticks(lags)
    ax.legend()#fontsize=9)
    if save_fig:
        fig.tight_layout()
        path = figures_dir / f"acf_{cond_safe}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if not _in_jupyter():
            plt.close(fig)
        print(f"  Saved: {path.name}")

    lag1_vals = acf_arr[:, 0]
    return dict(
        condition=condition,
        **{
            f"lag1_mean{suffix}": round(float(np.nanmean(lag1_vals)), 4),
            f"lag1_SD{suffix}":   round(float(np.nanstd(lag1_vals)),  4),
        }
    )


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX diagnostics summary table
# ═══════════════════════════════════════════════════════════════════════════

_DIAG_COL_LABELS: dict[str, str] = {
    "condition":             "Condition",
    "logZ_M0_mean":          r"$\overline{\log Z}$ M0",
    "logZ_M0_SD":            r"SD$_{\log Z}$ M0",
    "logZ_M1_mean":          r"$\overline{\log Z}$ M1",
    "logZ_M1_SD":            r"SD$_{\log Z}$ M1",
    "log_bf10":              r"$\log_{10} \mathrm{BF}_{10}$",
    "bf_SD_over_difference": r"SD$/|\Delta\log Z|$",
    "pB_mean_m0":            r"$p_B(\bar{y})$ M0",
    "pB_sd_m0":              r"$p_B(\mathrm{SD})$ M0",
    "pB_mean":               r"$p_B(\bar{y})$ M1",
    "pB_sd":                 r"$p_B(\mathrm{SD})$ M1",
    "residual_mean_m0":      r"Res.\ mean M0",
    "residual_SD_m0":        r"Res.\ SD M0",
    "shapiro_W_m0":          r"Shapiro $W$ M0",
    "shapiro_p_m0":          r"Shapiro $p$ M0",
    "residual_mean":         r"Res.\ mean M1",
    "residual_SD":           r"Res.\ SD M1",
    "shapiro_W":             r"Shapiro $W$ M1",
    "shapiro_p":             r"Shapiro $p$ M1",
    "lag1_mean_m0":          r"ACF lag-1 mean M0",
    "lag1_SD_m0":            r"ACF lag-1 SD M0",
    "lag1_mean":             r"ACF lag-1 mean M1",
    "lag1_SD":               r"ACF lag-1 SD M1",
    "max_rhat_m0":           r"$\hat{R}_{\max}$ M0",
    "max_rhat_m1":           r"$\hat{R}_{\max}$ M1",
}

# Decimal places per column, matched to the resolution the quantity actually
# has.  The log Z block stays at two decimals: those estimates carry a
# between-chain SD of 0.03-0.42 nats, so a third decimal would claim precision
# the SMC run does not have.  Everything else is a unit-scale quantity judged
# against a fixed criterion (0.1, 1.01, 0.5, 0.05, 0) that three decimals
# resolve comfortably.
_DIAG_COL_DECIMALS: dict[str, int] = {
    "logZ_M0_mean": 2, "logZ_M0_SD": 2,
    "logZ_M1_mean": 2, "logZ_M1_SD": 2,
    "log_bf10": 2, "bf_SD_over_difference": 3,
    "pB_mean_m0": 3, "pB_sd_m0": 3,
    "pB_mean": 3, "pB_sd": 3,
    "residual_mean_m0": 3, "residual_SD_m0": 3,
    "shapiro_W_m0": 3, "shapiro_p_m0": 3,
    "residual_mean": 3, "residual_SD": 3,
    "shapiro_W": 3, "shapiro_p": 3,
    "lag1_mean_m0": 3, "lag1_SD_m0": 3,
    "lag1_mean": 3, "lag1_SD": 3,
    "max_rhat_m0": 3, "max_rhat_m1": 3,
}

# Frequentist p-values: printed as "<0.001" below the table resolution, never as 0.
_DIAG_P_VALUE_COLS = {"shapiro_p_m0", "shapiro_p"}

_DIAG_GROUP_BF   = ["logZ_M0_mean", "logZ_M0_SD", "logZ_M1_mean", "logZ_M1_SD",
                    "log_bf10", "bf_SD_over_difference"]
_DIAG_GROUP_PPC  = ["pB_mean_m0", "pB_sd_m0", "pB_mean", "pB_sd"]
_DIAG_GROUP_RES  = ["residual_mean_m0", "residual_SD_m0", "shapiro_W_m0", "shapiro_p_m0",
                    "residual_mean", "residual_SD", "shapiro_W", "shapiro_p"]
_DIAG_GROUP_ACF  = ["lag1_mean_m0", "lag1_SD_m0", "lag1_mean", "lag1_SD"]
_DIAG_GROUP_RHAT = ["max_rhat_m0", "max_rhat_m1"]

# Row blocks of the combined diagnostics table, in print order.
_DIAG_GROUPS: list[tuple[str, list[str]]] = [
    ("Model comparison (SMC)", _DIAG_GROUP_BF),
    ("Convergence",            _DIAG_GROUP_RHAT),
    ("Posterior predictive",   _DIAG_GROUP_PPC),
    ("Residuals",              _DIAG_GROUP_RES),
    ("Autocorrelation",        _DIAG_GROUP_ACF),
]


def _write_grouped_diagnostics_table(
    table: pd.DataFrame,
    groups: list[tuple[str, list[str]]],
    output_path: Path,
    caption: str,
    note: str,
    use_sideways: bool = False,
) -> None:
    """Write one booktabs LaTeX table with diagnostics as rows, conditions as columns.

    *table* holds one row per condition with pre-formatted string values; it is
    transposed so that each diagnostic quantity becomes a row and each condition
    a column. *groups* supplies the row blocks as ``(header, columns)`` pairs;
    columns absent from *table* are skipped, and a block with no surviving
    column is dropped entirely.
    """
    conditions = list(table["condition"])
    n_cols = len(conditions) + 1
    col_spec = "l" + "r" * len(conditions)

    env = "sidewaystable" if use_sideways else "table"
    lines = [
        r"% Requires: \usepackage{booktabs} in preamble",
        rf"\begin{{{env}}}",
        r"  \small",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        r"  \label{{tab:{}}}".format(output_path.stem),
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        "    Diagnostic & " + " & ".join(conditions) + r" \\",
        r"    \midrule",
    ]

    first_group = True
    for header, group_cols in groups:
        present = [c for c in group_cols if c in table.columns]
        if not present:
            continue
        if not first_group:
            lines.append(r"    \addlinespace")
        first_group = False
        lines.append(
            rf"    \multicolumn{{{n_cols}}}{{@{{}}l}}{{\textit{{{header}}}}} \\"
        )
        for col in present:
            label = _DIAG_COL_LABELS.get(col, col)
            values = " & ".join(str(v) for v in table[col])
            lines.append(rf"    \quad {label} & {values} \\")

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \par\smallskip",
        rf"  {{\footnotesize\textit{{Note.}} {note}}}",
        rf"\end{{{env}}}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {output_path}")


def save_diagnostics_latex_table(
    summary_df: pd.DataFrame,
    output_path: Path,
    parameter_col: str = "",
) -> None:
    """Write one combined booktabs LaTeX table covering every diagnostic group.

    The table is written to *output_path* with diagnostics as rows (grouped into
    the blocks of :data:`_DIAG_GROUPS`) and chromatic conditions as columns, so
    that all diagnostics for one outcome fit on a single page. Requires
    ``booktabs`` in the preamble.
    """
    from civibeplr.analysis_io import PARAMETER_TITLE, lowercase_first_word
    param_label = lowercase_first_word(PARAMETER_TITLE.get(parameter_col, parameter_col))

    table = summary_df.copy()
    if "condition" in table.columns:
        table["condition"] = (
            table["condition"].astype(str).str.lower()
            .replace({"l-m": "lm"})
            .apply(lambda c: rf"\acs{{{c}}}")
        )
    table = table.sort_values("condition")

    # Columns with a pass/fail criterion: value fails when it is >= (or <=,
    # for the R-hat direction flipped) the threshold, and is flagged bold so
    # it is visible at a glance in the printed table.
    _BOLD_THRESHOLDS: dict[str, tuple[float, str]] = {
        "bf_SD_over_difference": (0.1, ">="),
        "max_rhat_m0":           (1.01, ">"),
        "max_rhat_m1":           (1.01, ">"),
    }

    for col, decimals in _DIAG_COL_DECIMALS.items():
        if col not in table.columns:
            continue
        if col in _DIAG_P_VALUE_COLS:
            table[col] = table[col].apply(format_p_value, decimals=decimals)
            continue
        fmt = f"{{:.{decimals}f}}"
        threshold = _BOLD_THRESHOLDS.get(col)

        def _fmt(value, f=fmt, threshold=threshold) -> str:
            if pd.isna(value):
                return "--"
            text = f.format(value)
            # A small negative value that rounds to zero prints as "-0.000",
            # which reads as an error rather than as "indistinguishable from 0".
            text = text[1:] if text.startswith("-") and float(text) == 0 else text
            if threshold is not None:
                limit, op = threshold
                fails = value >= limit if op == ">=" else value > limit
                if fails:
                    text = rf"\textbf{{{text}}}"
            return text

        table[col] = table[col].apply(_fmt)

    caption = (
        f"Model diagnostics for {param_label} per chromatic condition."
    )
    note = (
        r"M0 is the null model, M1 the circadian cosinor model; quantities are "
        r"reported for both wherever they are defined for each model separately. "
        r"\emph{Model comparison}: $\overline{\log Z}$ is the mean SMC log marginal "
        r"likelihood; SD$/|\Delta\log Z|$ is the pooled SD of $\log Z$ across the "
        r"independent SMC populations divided by the absolute $\log Z$ difference, "
        r"and values $\geq 0.1$ (bold) mark a numerically imprecise Bayes factor. "
        r"\emph{Convergence}: $\hat{R}_{\max}$ is the maximum $\hat{R}$ across all "
        r"parameters, computed across independent SMC populations; it detects "
        r"disagreement between populations, not within-run mixing, and values "
        r"$> 1.01$ are shown in bold. "
        r"\emph{Posterior predictive}: $p_B$ is the Bayesian $p$-value for the "
        r"posterior predictive mean and SD; values near 0.5 indicate good fit. "
        r"\emph{Residuals}: mean and SD of pooled posterior-mean residuals; "
        r"Shapiro-Wilk $W$ and $p$ test residual normality. "
        r"\emph{Autocorrelation}: mean and SD across participants of the "
        r"within-participant residual ACF at a lag of one block (3.75~h); pairs "
        r"spanning an excluded block are not used. This is descriptive: with "
        r"roughly ten blocks per participant the lag-1 ACF of uncorrelated residuals is "
        r"expected near $-1/n$ rather than 0, and a mean near 0 can arise from "
        r"participant ACFs of opposite sign (see SD)."
    )

    _write_grouped_diagnostics_table(
        table, _DIAG_GROUPS, output_path, caption, note
    )


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

def _match_condition(cond_safe: str, all_conds: list[str]) -> str | None:
    """Match a file-safe condition string back to its original label."""
    for c in all_conds:
        if c.replace(" ", "_").lower() == cond_safe:
            return c
    for c in all_conds:
        if c.replace(" ", "").lower() == cond_safe.replace("_", "").lower():
            return c
    return None


def run_diagnostics(
    conditions: list[str] | None = None,
    parameter_col: str = "amplitude",
    idata_dir: Path | None = None,
    figures_dir: Path | None = None,
    tables_dir: Path | None = None,
    tex_dir: Path | None = None,
) -> pd.DataFrame:
    """Run all eight diagnostics for every condition and save outputs.

    If *conditions* is None all ``*_M1.nc`` files in *idata_dir* are used.

    Returns
    -------
    pd.DataFrame
        One row per condition with all numerical diagnostic quantities.
    """
    if idata_dir   is None: idata_dir   = DEFAULT_IDATA_DIR / parameter_col
    if figures_dir is None: figures_dir = DEFAULT_DIAG_FIGURES_DIR / parameter_col
    if tables_dir  is None: tables_dir  = DEFAULT_DIAG_TABLES_DIR / parameter_col
    if tex_dir     is None: tex_dir     = DEFAULT_DIAG_TEX_DIR / parameter_col

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    tex_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover conditions from saved files ──────────────────────────────
    if conditions is None:
        m1_files = sorted(idata_dir.glob("*_M1.nc"))
        if not m1_files:
            raise FileNotFoundError(
                f"No *_M1.nc files found in {idata_dir}.\n"
                "Run analyse_circadian_pupil_parameters.py --fit first."
            )
        conditions_safe = [f.stem.replace("_M1", "") for f in m1_files]
        # Recover original condition names from saved data CSVs
        conditions_orig = []
        for cs in conditions_safe:
            csv = idata_dir / f"{cs}_data.csv"
            if csv.exists():
                df_tmp = pd.read_csv(csv, nrows=1)
                if PUPIL_CONDITION_COL in df_tmp.columns:
                    conditions_orig.append(str(df_tmp[PUPIL_CONDITION_COL].iloc[0]))
                else:
                    conditions_orig.append(cs)
            else:
                conditions_orig.append(cs)
        conditions = conditions_orig
        print(f"Auto-detected conditions: {conditions}")

    # ════════════════════════════════════════════════════════════════════════
    # Multi-panel log Z figure (one subplot per condition)
    # ════════════════════════════════════════════════════════════════════════
    n_cond  = len(conditions)
    ncols_lz = min(3, n_cond)
    nrows_lz = int(np.ceil(n_cond / ncols_lz))
    fig_lz, axes_lz = plt.subplots(
        nrows_lz, ncols_lz,
        figsize=(5 * ncols_lz, 4 * nrows_lz),
        squeeze=False,
    )
    fig_lz.suptitle("SMC log Z per chain: BF reliability")#, fontsize=13)

    summary_rows: list[dict] = []

    for ci, condition in enumerate(conditions):
        print(f"\n── Condition: {condition.upper()} ──────────────────")

        idata_m0, idata_m1 = load_idata(condition, idata_dir)
        df_cond = load_condition_data(condition, idata_dir)

        row: dict = {"condition": condition}

        # 1 — log Z variance (into shared multi-condition figure)
        ax_lz = axes_lz[ci // ncols_lz][ci % ncols_lz]
        lz_stats = check_logz_variance(idata_m0, idata_m1, condition, ax=ax_lz)
        row.update({k: v for k, v in lz_stats.items() if k != "condition"})

        # 4 — Bayesian p-value (no plot)
        pb_stats = compute_bayesian_pvalue(idata_m1, df_cond, condition, parameter_col)
        row.update({k: v for k, v in pb_stats.items() if k != "condition"})
        print(f"  Bayesian p-value: p_B(mean)={pb_stats['pB_mean']:.3f}  "
              f"p_B(SD)={pb_stats['pB_sd']:.3f}")

        # 5 — Posterior parameter summary (table only)
        summarise_posterior(idata_m1, condition, parameter_col,
                            tables_dir=tables_dir)

        # ── Combined 3×3 diagnostic figure for this condition ────────────
        # Layout (row × col):
        #   [0,0] prior predictive   [0,1] PPC density       [0,2] PPC scatter
        #   [1,0] shrinkage          [1,1] residuals hist    [1,2] residuals Q-Q
        #   [2,0] ACF                [2,1] residuals fitted  [2,2] residuals time
        cond_safe = condition.replace(" ", "_").lower()
        fig_cond, axs = plt.subplots(3, 3, figsize=(12.5, 11), squeeze=False)
        fig_cond.suptitle(f"Diagnostics on {condition.upper()} on\n{lowercase_first_word(get_parameter_label(parameter_col, title=True))}", y=1.01) #fontsize=14, 

        # 2 — Prior predictive
        plot_prior_predictive(
            idata_m1, df_cond, condition, parameter_col, ax=axs[0, 0]
        )

        # 3 — PPC
        plot_ppc(idata_m1, df_cond, condition, parameter_col,
                 ax_density=axs[0, 1], ax_scatter=axs[0, 2])

        # 6 — Shrinkage
        plot_shrinkage(idata_m1, df_cond, condition, parameter_col, ax=axs[1, 0])

        # 7 — Residuals (4 panels: [1,1], [1,2], [2,0], [2,1])
        resid_stats = plot_residuals(
            idata_m1, df_cond, condition, parameter_col,
            axes=[axs[1, 1], axs[1, 2], axs[2, 1], axs[2, 2]],
        )
        row.update({k: v for k, v in resid_stats.items() if k != "condition"})

        # 8 — ACF
        acf_stats = plot_acf_within_participant(
            idata_m1, df_cond, condition, parameter_col, ax=axs[2, 0]
        )
        row.update({k: v for k, v in acf_stats.items() if k != "condition"})

        # 8b — Same PPC, residual and ACF diagnostics for the null model M0.
        # The reported evidence favours M0 in most conditions, so the adequacy of
        # M0 itself has to be checkable, not just that of M1. Comparing residual SD
        # and lag-1 ACF between the two models also shows how much structure the
        # cosinor regressors actually absorb.
        #
        # plot_residuals and plot_acf_within_participant compute and plot in one
        # call, so they are handed a scratch figure that is discarded immediately:
        # the published diagnostics figure stays M1-only.
        row.update(
            {
                k: v
                for k, v in compute_bayesian_pvalue(
                    idata_m0, df_cond, condition, parameter_col, suffix="_m0"
                ).items()
                if k != "condition"
            }
        )

        fig_scratch, axs_scratch = plt.subplots(2, 3, figsize=(1, 1), squeeze=False)
        try:
            m0_resid = plot_residuals(
                idata_m0, df_cond, condition, parameter_col,
                axes=[axs_scratch[0, 0], axs_scratch[0, 1],
                      axs_scratch[0, 2], axs_scratch[1, 0]],
                suffix="_m0",
            )
            m0_acf = plot_acf_within_participant(
                idata_m0, df_cond, condition, parameter_col,
                ax=axs_scratch[1, 1], suffix="_m0",
            )
        finally:
            plt.close(fig_scratch)
        row.update({k: v for k, v in m0_resid.items() if k != "condition"})
        row.update({k: v for k, v in m0_acf.items() if k != "condition"})

        # 9 — r_hat convergence
        max_rhat_m0, max_rhat_m1 = load_rhat_summary(
            condition, idata_dir=idata_dir, idata_m0=idata_m0, idata_m1=idata_m1
        )
        row["max_rhat_m0"] = max_rhat_m0
        row["max_rhat_m1"] = max_rhat_m1
        flag = lambda v: " [!]" if not np.isnan(v) and v > 1.01 else ""
        print(f"  r_hat  M0 max={max_rhat_m0:.4f}{flag(max_rhat_m0)}"
              f"  M1 max={max_rhat_m1:.4f}{flag(max_rhat_m1)}")

        panel_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
        for idx, ax in enumerate(axs.flat):
            ax.text(
                -0.12, 1.2, panel_labels[idx],
                transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left",
            )

        fig_cond.tight_layout()
        cond_path = figures_dir / f"fig_{parameter_col.lower()}_diagnostics_{cond_safe}.png"
        fig_cond.savefig(cond_path, dpi=150, bbox_inches="tight")
        if not _in_jupyter():
            plt.close(fig_cond)
        print(f"  Saved: {cond_path.name}")

        summary_rows.append(row)

    # ── Finalise log Z figure ─────────────────────────────────────────────
    for ci in range(n_cond, nrows_lz * ncols_lz):
        axes_lz[ci // ncols_lz][ci % ncols_lz].set_visible(False)
    fig_lz.tight_layout()
    lz_path = figures_dir / "logz_variance_all_conditions.png"
    fig_lz.savefig(lz_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {lz_path.name}")
    if not _in_jupyter():
        plt.close(fig_lz)

    # ── Summary table ─────────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    out_path = tables_dir / "diagnostics_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(summary_df.to_string(index=False))

    latex_path = tables_dir / f"tab_diagnostics_summary_{parameter_col}.tex"
    save_diagnostics_latex_table(summary_df, latex_path, parameter_col=parameter_col)

    # Namespace the macro prefix by metric, as save_results() does for the cosinor
    # parameter vars: this function runs once per outcome metric, so a shared prefix
    # makes every metric emit the same \def names. Inputting two of the resulting
    # files silently redefines the earlier metric's values and the manuscript then
    # cites the wrong model's diagnostics.
    # p-value macros use the same formatting as the table, so text and table agree.
    vars_df = summary_df.copy()
    for col in _DIAG_P_VALUE_COLS & set(vars_df.columns):
        vars_df[col] = vars_df[col].apply(format_p_value, decimals=_DIAG_COL_DECIMALS[col])
    tex_vars = format_latex_variables(
        vars_df, row_key_col="condition", prefix=f"diagnose_circadian_{parameter_col}"
    )
    vars_path = tex_dir / f"vars_diagnose_circadian_{parameter_col}.tex"
    vars_path.write_text(tex_vars, encoding="utf-8")
    print(f"Saved LaTeX variables: {vars_path}")

    return summary_df


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Circadian pupil model diagnostics."
    )
    parser.add_argument(
        "--parameter", default="amplitude",
        help="Parameter column (default: amplitude)",
    )
    parser.add_argument(
        "--idata-dir", type=Path, default=None,
        help="Directory containing *_M0.nc / *_M1.nc / *_data.csv files.",
    )
    args = parser.parse_args()

    run_diagnostics(
        parameter_col=args.parameter, 
        idata_dir=args.idata_dir
        )


if __name__ == "__main__":
    main()
