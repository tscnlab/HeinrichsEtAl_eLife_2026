
<img src="https://github.com/tscnlab/Templates/blob/main/logo/logo_with_text-01.png" width="400"/>

# Supplementary Material for Heinrichs & Spitschan, *No detectable circadian modulation of photoreceptor-directed pupil responses during an ultra-short sleep-wake protocol* 

## Description 

Analysis pipeline for pupillary light responses to sinusoidal chromatic stimulation, recorded with a RetinaWISE pupillometer (Oculox Technologies). Data was collected during wake periods of 40-hour forced desynchrony protocol with an ultra-short sleep-wake rhythm, cf. Heinrichs & Spitschan (2025, PlosONE).
The pipeline covers the full path from raw per-participant recordings to cosinor-modelled parameters, summary metrics, and publication-ready figures and tables.

## Full reference

If you use this code or data, please cite our article:

> Heinrichs, H.S., & Spitschan, M. (2026). No detectable circadian modulation of photoreceptor-directed pupil responses during an ultra-short sleep-wake protocol. *bioRxiv*. [https://doi.org/xxx](https://doi.org/xxx)

## Contact

For any questions or comments, get in touch via email to Manuel Spitschan ([manuel.spitschan@tum.de](mailto:manuel.spitschan@tum.de)).

## License

**Supplementary Material for Heinrichs & Spitschan, *No detectable circadian modulation of photoreceptor-directed pupil responses during an ultra-short sleep-wake protocol*** is released under the MIT License.

© 2026 Hannah Heinrichs and co-authors. See https://opensource.org/licenses/MIT for the full license text.


## Project overview

Participants viewed sinusoidal light stimuli at **0.4 Hz** across five chromatic conditions — **melanopsin (Mel)**, **S-cone (S)**, **L–M cone-opponent (L-M)**, **luminance (LMS)**, and **broadband flux (LMS+Mel)** — over multiple blocks. Pupil size was recorded monocularly from the dominant eye (determined by Miles test) and processed through:

1. **Data preparation** — Preprocessing of data saved by the RetinaWise device. Code and full description of preprocessing pipeline available in [pupil-prep](https://github.com/tscnlab/civibe_pupilprep); cleaned per-participant CSVs of different preprocessing stages saved in `./data/retinwise_results`
1. **Aggregation** — Per-trial traces averaged across tests, baseline-corrected, and flux-normalised (Python).
1. **Descriptive metrics** — Phase-specific means, constriction amplitude, AUC (Python).
1. **Two-step cosinor modelling** — Low-pass filter separates a slow *tonic* constriction (Step 1) from a fast *sinusoidal tracking* residual fitted with a fundamental + 2nd-harmonic cosinor (Step 2).
1. **Saliva cosinor model** — Per-participant circadian period τ estimated from melatonin concentrations by scanning a two-harmonic cosinor over τ = 22–28 h and selecting the best fit by AIC.
1. **Circadian pupil analysis** — Bayesian mixed-effects cosinor tests whether log(amplitude) from the two-step model varies with a near-circadian rhythm across blocks, using participant-specific τ from saliva data.

## Repository structure

```
civibe_pupillometry/
├── data/
│   ├── retinawise_results/     # Per-participant data from RetinaWise
│   │   ├── raw/                # Unprocessed exports
│   │   ├── cleaned/            # After QC / artefact removal
│   │   ├── complete/           # Complete trials only
│   │   └── final/              # Ready for analysis (12 participants)
│   ├── aggregated/             # Combined, baseline-corrected dataset
│   │   ├── data.csv
│   │   └── metadata.csv
│   └── raw/                    # Other raw data (e.g. saliva)
│       └── clean_saliva_data.csv
│   
├── src/civibeplr/              # Python package
│   │   # -- shared helpers (imported, not run directly) --
│   ├── data_io.py              # Shared data-loading helpers & column constants
│   ├── analysis_io.py          # Shared analysis constants (stimulus window timings)
│   ├── vis_io.py               # Shared plotting helpers (figure format, fonts, colours)
│   ├── utils_cosinor.py        # Cosinor math helpers (amplitude/phase back-transforms)
│   ├── latex_table_creation.py # LaTeX table / variable macros generation
│   ├── load_final_data.py      # Load individual or all participant CSVs
│   │   
│   │   # --Pipeline steps (run via pipeline.py or python -m; see table below) --
│   ├── get_participant_info.py                       # Step 0: demographics/questionnaire export
│   ├── aggregate_pupil_traces.py                     # Step 1: aggregate, baseline-correct, flux-normalise
│   ├── calculcate_descriptive_pupil_metrices.py      # Step 2: descriptive pupillary metrics
│   ├── fit_pupil_trace_2step_cosinor_model.py        # Step 3: two-step cosinor (LP filter + cosinor)
│   ├── visualise_pupil_cosinor.py                    # Step 3b: cosinor result / decomposition figures
│   ├── fit_saliva_cosinor_model.py                   # Step 4: cosinor fit to saliva melatonin data
│   ├── analyse_circadian_pupil_parameters.py         # Step 5: Bayesian circadian modulation test
│   ├── diagnose_circadian_models.py                  # Step 5b: post-fit diagnostics for Step 5 models
│   ├── visualise_circadian_parameter_cosinor.py      # Step 5c: post-hoc fitted-vs-empirical figures
│   ├── diagnose_pupil_models.py                      # Post-fit diagnostics for Step 3 (tonic scatter, R^2)
│   │   
│   │   # -- Supplementary analysis only (run via si_pipeline.py; see below) --
│   ├── si_derive_protocol_plr.py                     # SI Step 3: derive the protocol-specified ratio R
│   ├── si_visualise_protocol_plr.py                  # SI Steps 3b/5c: endpoint, ordering, ratio and BF figures
│   ├── si_analyse_plr_ratio.py                       # SI Steps 4b/5/5b: inspect, fit and diagnose M0/M1 on R
│   ├── si_sensitivity_protocol_plr.py                # SI Step 6: leave-one-out (L-M) and registered tau range
│   │   
│   │   # -- NOT part of the analysis pipeline; nothing below feeds the manuscript --
│   │   #    or the reported figures/tables. Exploratory / inspection / abstract-art only.
│   ├── visualise_circadian_pupil_overview.py     # Exploratory 3-panel circadian overview figure
│   ├── inspect_circadian_model.py                    # Ad-hoc circadian InferenceData inspection
│   ├── plot_protocol_abstract.py                     # Graphical-abstract protocol timeline
│   └── not_used/             # Deprecated / exploratory scripts
├── misc/                       # Explanatory / comparison plots
├── results/
│   ├── tables/                 # Generated CSV and LaTeX tables
│   │   ├── pupil_cosinor/
│   │   ├── saliva_cosinor/
│   │   └── circadian_cosinor/
│   └── figures/                # Generated figures
│       ├── pupil_cosinor/
│       └── saliva_cosinor/
├── notebooks/                  # Jupyter notebooks (exploratory)
├── doc/                        # Documentation (placeholder)
├── reports/                    # Reports (placeholder)
├── tests/                      # pytest tests (test_si_* cover the supplementary analysis)
├── pipeline.py / .ipynb        # Runs the main pipeline steps in order
├── si_pipeline.py / .ipynb     # Runs the supplementary-analysis steps in order
├── pyproject.toml
├── LICENSE                     # MIT
└── README.md
```

## Installation

The Python package is installable in editable mode. A **conda** environment is recommended. The specification of the used conda environment `civibe_pupillometry` will be made available in a YAML.

```bash
# Create and activate environment
conda create -n civibeplr python=3.11 -y
conda activate civibeplr

# Install the package in editable mode (pulls in dependencies)
pip install -e .
```

### Dependencies

- Python ≥ 3.9
- numpy, pandas, scipy, statsmodels, matplotlib, seaborn
- bambi, pymc (Steps 4–5 only)

---

## Usage

The scripts are designed to run sequentially. Each step reads the output of the previous one.

### Prerequisite — Preprocessing data

> Preprocess raw RetinaWise exports into cleaned, per-participant CSVs.
> Part of a separate repository, run before this pipeline: https://github.com/tscnlab/civibe_pupilprep


**Output:** `data/retinawise_results/final/<ID>_final_data.csv`


### Step 0 — Participant demographics

> Load lab/online screening and session info; compute age at study entry, BMI, PSQI total, and MCTq chronotype (MSF_sc); write demographics CSV and LaTeX tables.

```bash
python -m civibeplr.get_participant_info
```

**Output:** `results/tables/participant_info/` (`participant_demographics.csv`, `tab_participant_demographics_summary.tex`, ...)


### Step 1 — Aggregate pupil traces

> Combine all participants, average across tests within condition × block, baseline-correct to pre-stimulus mean, and compute flux-normalised ratios.

```bash
python -m civibeplr.aggregate_pupil_traces
```

**Input:** `data/retinawise_results/final/`

**Output:**
- `data/aggregated/data.csv` — combined dataset with the following key columns:
  - `pupil_mean` — mean pupil diameter averaged across tests (mm)
  - `pupil_std` — standard deviation across tests (mm)
  - `baseline_mm` — mean pupil diameter during the pre-stimulus phase (mm)
  - `rel_size_mm` — baseline-corrected pupil diameter (`pupil_mean − baseline_mm`, mm)
  - `change_size_pct` — percentage change from baseline (`(pupil_mean − baseline_mm) / baseline_mm × 100`, %)
- `data/aggregated/metadata.csv` — column descriptions
- `results/figures/means_across_blocks.png`

<details>
<summary>CLI options</summary>

| Flag | Default | Description |
|---|---|---|
| `--data-dir` | `data/retinawise_results/final/` | Input directory |
| `--output` | `data/aggregated/data.csv` | Output CSV path |

</details>

### Step 2 — Calculate descriptive metrics

> Compute per-group summary statistics: phase-specific means, maximum constriction amplitude, and area under the curve (AUC) during the stimulus window and post-stimulus phases. The stim/poststim boundary is controlled by `DEFAULT_ADD_POSTSTIM_SEC` in `analysis_io.py` (default: 0, i.e. stimulus window only).

```bash
python -m civibeplr.calculcate_descriptive_pupil_metrices
```

**Input:** `data/aggregated/data.csv`

**Output:**
- `results/descriptives/pupil_metrics.csv` — per-group summary metrics
- `results/descriptives/pupil_metrics_metadata.csv` — column descriptions
- `results/descriptives/pupil_metrics.tex` — full LaTeX table
- `results/descriptives/tab_pupil_descriptive_metrics_selected.tex` — selected metrics for manuscript

<details>
<summary>CLI options</summary>

| Flag | Default | Description |
|---|---|---|
| `--data-path` | `data/aggregated/data.csv` | Aggregated data CSV |
| `--output` | `results/descriptives/pupil_metrics.csv` | Output CSV path |

</details>

### Step 3 — Two-step cosinor model

> **Step 1:** Low-pass Butterworth filter (default cutoff 0.2 Hz) separates a slow tonic constriction.
>
> **Step 2:** OLS cosinor (fundamental f + 2nd harmonic 2f) on the residual during the analysis window (stimulus window by default; extend with `--add-poststim-sec` or `DEFAULT_ADD_POSTSTIM_SEC`).
>
> Produces tonic features (min, mean), sinusoidal amplitudes, phases, and goodness-of-fit statistics. Phase averaging uses circular statistics. The ratio of fundamental to 2nd-harmonic amplitude (`amplitude_ratio = amplitude / amplitude_2f`) is also reported as a measure of waveform linearity; NaN when the 2nd-harmonic amplitude is zero.

```bash
python -m civibeplr.fit_pupil_trace_2step_cosinor_model
```

**Input:** `data/aggregated/data.csv`

**Output:**
- `results/tables/pupil_cosinor/pupil_cosinor_2step_param.csv`
- `results/figures/pupil_cosinor/fig_pupil_cosinor_parameters.png`
- `results/figures/pupil_cosinor/fig_raincloud_parameters.png`
- `results/figures/pupil_cosinor/fig_param_changes_<param>.png` (one per parameter)
- `results/figures/pupil_cosinor/fig_pupil_cosinor_parameters_<ID>.png` (exemplary participant)
- `results/figures/pupil_cosinor/fig_pupil_cosinor_decomposition_average.png`
- `results/figures/pupil_cosinor/fig_pupil_cosinor_decomposition_<ID>.png` (exemplary participant)
- `results/figures/pupil_cosinor/participant_<ID>/fig_cosinor_2step_fits_with_overlay_<ID>.png` (per participant)

<details>
<summary>CLI options</summary>

| Flag | Default | Description |
|---|---|---|
| `--data` | `data/aggregated/data.csv` | Aggregated data CSV |
| `--outcome` | `rel_size_mm` | Outcome column |
| `--freq` | `0.4` | Stimulus frequency (Hz) |
| `--cutoff` | `0.2` | Low-pass cutoff (Hz) |
| `--filter-order` | `2` | Butterworth filter order |
| `--add-poststim-sec` | `0` | Post-stimulus seconds appended to fit window (default: stimulus window only) |
| `--output-dir` | `results/` | Output directory |

</details>

### Step 3b — Diagnose pupil models (optional)

> Diagnostic plots for the two-step cosinor fit. Loads the fitted parameters CSV from Step 3 and generates a tonic scatter plot (Step 1) and an R² distribution plot (Step 2). Requires Step 3 to have been run first.

```bash
python -m civibeplr.diagnose_pupil_models
```

**Input:** `results/tables/pupil_cosinor/pupil_cosinor_2step_param.csv`

**Output:**
- `results/figures/pupil_cosinor/fig_pupil_cosinor_tonic_scatter.png`
- `results/figures/pupil_cosinor/fig_pupil_cosinor_r_squared_distribution.png`

<details>
<summary>CLI options</summary>

| Flag | Default | Description |
|---|---|---|
| `--results` | `results/tables/pupil_cosinor/pupil_cosinor_2step_param.csv` | Fitted parameters CSV |
| `--output-dir` | `results/figures/pupil_cosinor/` | Output directory for figures |

</details>

### Step 3c — Visualise pupil cosinor (optional)

> Generate parameter summary figures and two-step decomposition plots from the fitted cosinor results. Parameter plots require only the results CSV; decomposition plots additionally require the aggregated data CSV and a settings JSON (saved during Step 3 via `--save-settings-json`) to guarantee identical filter settings.

```bash
# Parameter plots only
python -m civibeplr.visualise_pupil_cosinor --results-csv results/tables/pupil_cosinor/pupil_cosinor_2step_param.csv

# Parameter + decomposition plots
python -m civibeplr.visualise_pupil_cosinor \
    --results-csv results/tables/pupil_cosinor/pupil_cosinor_2step_param.csv \
    --data data/aggregated/data.csv \
    --load-settings-json <path/to/settings.json>
```

**Input:**
- `results/tables/pupil_cosinor/pupil_cosinor_2step_param.csv` (Step 3)
- `data/aggregated/data.csv` (decomposition plots only)
- Settings JSON saved with `--save-settings-json` during Step 3 (decomposition plots only)

**Output:**
- `results/figures/pupil_cosinor/fig_pupil_cosinor_parameters.png` — all parameters × conditions × blocks
- `results/figures/pupil_cosinor/fig_raincloud_parameters.png` — raincloud grid for all parameters
- `results/figures/pupil_cosinor/fig_param_changes_<param>.png` — per-parameter dynamics + raincloud
- `results/figures/pupil_cosinor/fig_pupil_cosinor_parameters_<ID>.png` — exemplary participant
- `results/figures/pupil_cosinor/fig_pupil_cosinor_decomposition_average.png` — grand-average tonic + cosinor overlay
- `results/figures/pupil_cosinor/fig_pupil_cosinor_decomposition_<ID>.png` — exemplary participant decomposition
- `results/figures/pupil_cosinor/participant_<ID>/fig_cosinor_2step_fits_with_overlay_<ID>.png` — per-participant (with `--plot-individual`)

<details>
<summary>CLI options</summary>

| Flag | Default | Description |
|---|---|---|
| `--results-csv` | `results/tables/pupil_cosinor/pupil_cosinor_2step_param.csv` | Fitted parameters CSV |
| `--data` | `data/aggregated/data.csv` | Aggregated data CSV (decomposition plots) |
| `--output-dir` | `results/` | Output directory |
| `--load-settings-json` | — | JSON with filter/model settings used during fitting (required for decomposition) |
| `--save-settings-json` | — | Save current decomposition settings to JSON |
| `--plot-individual` | — | Generate per-participant decomposition plots (slow) |
| `--exemplary` | `2` | 0-based participant index used for exemplary plots |

</details>

### Step 4 — Saliva cosinor model

> Fit a two-harmonic cosinor to per-participant melatonin concentration data. The period τ is scanned over 22–28 h (1-minute steps); best τ per participant is selected by AIC.

```bash
python -m civibeplr.fit_saliva_cosinor_model
```

**Input:** `data/raw/clean_saliva_data.csv`

**Output:**
- `results/tables/saliva_cosinor/saliva_cosinor_best_fit.csv` — best-fit τ and parameters per participant
- `results/tables/saliva_cosinor/saliva_cosinor_best_fit.tex` — LaTeX summary table
- `results/figures/saliva_cosinor/fig_best_fits_per_participant.png`
- `results/figures/saliva_cosinor/fig_metric_vs_tau.png`
- `results/figures/saliva_cosinor/fig_saliva_cosinor_summary.png`

<details>
<summary>CLI options</summary>

| Flag | Default | Description |
|---|---|---|
| `--data-path` | `data/raw/clean_saliva_data.csv` | Input saliva CSV |
| `--outcome-col` | `melatonin_concentration` | Outcome column |
| `--time-col` | `standard_datetime` | Datetime column |
| `--period-min` | `22.0` | Minimum period to test [h] |
| `--period-max` | `28.0` | Maximum period to test [h] |
| `--period-step` | `1/60` | Period step [h] |
| `--selection-metrics` | `aic` | Metric(s) for τ selection |
| `--output-dir` | `results/` | Output directory |
| `--no-plot` | — | Skip plot generation |

</details>

### Step 5 — Circadian pupil analysis

> Test whether log(amplitude) from the two-step cosinor varies with a near-circadian rhythm across blocks, using participant-specific τ from saliva data. Fits a null model (M0: intercept + random intercept) and a cosinor model (M1: cosinor regressors + random slopes) per chromatic condition via bambi/pymc SMC sampling. Bayes factor BF₁₀ is computed from log marginal likelihoods.

```bash
python -m civibeplr.analyse_circadian_pupil_parameters --fit
```

**Input:**
- `results/tables/pupil_cosinor/pupil_cosinor_2step_param.csv`
- `results/tables/saliva_cosinor/saliva_cosinor_best_fit.csv`

**Output:** `results/tables/circadian_cosinor/`

<details>
<summary>CLI options</summary>

| Flag | Default | Description |
|---|---|---|
| `--fit` | — | Run SMC sampling (required to fit models) |
| `--smc-draws` | `10000` | Number of SMC draws; do not lower (too few particles bias log Z) |
| `--seed` | `1072` | Random seed |
| `--output-dir` | `results/tables/circadian_cosinor/` | Output directory |

</details>


### Step 5b — Diagnose circadian models (optional)

> Run eight diagnostic checks on the fitted SMC models from Step 5.

```bash
python -m civibeplr.diagnose_circadian_models
```

**Input:** `data/derivatives/circadian_cosinor/` (InferenceData NetCDF files from Step 5)

| # | Check | What it detects |
|---|-------|----------------|
| 1 | **SMC log-Z variance** | Unreliable Bayes factor: high SD of log Z across chains relative to the BF difference |
| 2 | **Prior predictive check** | Priors too tight or too diffuse relative to the observed data range |
| 3 | **Posterior predictive check (PPC)** | Overall model mis-fit: simulated data distribution vs. observed |
| 4 | **Bayesian p-value** | Systematic bias (`p_B(mean)`) or variance mis-specification (`p_B(SD)`); values near 0 or 1 indicate problems |
| 5 | **Posterior parameter summary** | Fixed-effect posterior means, SDs, and 94 % HDIs for M1 |
| 6 | **Random-effects shrinkage** | Partial-pooling vs. no-pooling per participant; extreme shrinkage flags influential observations |
| 7 | **Residual analysis** | Non-normality, heteroscedasticity, and temporal trend in residuals (4-panel: histogram, Q-Q, vs. fitted, vs. time) |
| 8 | **Within-participant ACF** | Residual autocorrelation across blocks; mean ACF ± 95 % bootstrap CI |
| 9 | **R-hat convergence** | Disagreement between independently-initialized SMC "chains" (particle populations, not sequential MCMC chains); flags multimodality, particle degeneracy, or seed-dependent bias, not within-chain mixing |

**Output:**
- `results/figures/circadian_cosinor_diagnostics/<outcome>/logz_variance_all_conditions.png`
- `results/figures/circadian_cosinor_diagnostics/<outcome>/diagnostics_<condition>.png` (combined 3x3 panel: checks 2-8)
- `results/tables/circadian_cosinor_diagnostics/<outcome>/diagnostics_summary.csv`
- `results/tables/circadian_cosinor_diagnostics/<outcome>/posterior_params_<condition>.csv`

<details>
<summary>CLI options</summary>

| Flag | Default | Description |
|---|---|---|
| `--parameter` | `amplitude` | Parameter column (outcome variable) |
| `--idata-dir` | `data/derivatives/circadian_cosinor/` | InferenceData directory |

</details>

### Step 5c — Visualise circadian cosinor fits (optional)

> Generate post-hoc fitted-vs-empirical plots for the circadian cosinor models. For each chromatic condition, overlays the M1 model posterior predictions on observed values of the selected outcome parameter (e.g. `amplitude` or `tonic_min`), both per participant and at the group level. Reads from `*_predictions.csv` files saved by Step 5 — no re-sampling required.
>
> By default this step produces the combined all-conditions figure for `amplitude` and `tonic_min`. To generate only the combined figure for these two metrics, run:
>
> ```bash
> python -m civibeplr.visualise_circadian_parameter_cosinor --layout combined
> ```

```bash
python -m civibeplr.visualise_circadian_parameter_cosinor
```

**Input:** `data/derivatives/circadian_cosinor/<outcome>/*_predictions.csv` (from Step 5)

**Output (per outcome):**
- `results/figures/circadian_cosinor/<outcome>/fig_fitted_vs_empirical_all_conditions.png` — all conditions in one figure (group-level fit, 95% HDI, block means, and per-participant trajectories)
- `results/figures/circadian_cosinor/<outcome>/fig_fitted_vs_empirical_<condition>.png` — detailed per-participant grid for one chromatic condition (when `--layout` is `both` or `per_condition`)

<details>
<summary>CLI options</summary>

| Flag | Default | Description |
|---|---|---|
| `--outcome` | `amplitude tonic_min` | One or more outcome variables to plot |
| `--layout` | `both` | `combined` (all-conditions figure only), `per_condition` (one figure per condition), or `both` |
| `--idata-dir` | `data/derivatives/circadian_cosinor/<outcome>/` | Directory with *_predictions.csv files |
| `--figures-dir` | `results/figures/circadian_cosinor/<outcome>/` | Output directory for figures |
| `--share-y` | — | Use a common y-axis across condition panels in the combined figure |
| `--individual-style` | `observed` | Draw per-participant traces as observed points (`observed`) or model fits (`fit`) |

</details>

## Supplementary analysis -- protocol-specified outcome

The published study protocol registered a different primary pupil outcome from the
one analysed in the main article: the mean trial-level maximum constriction for each
condition, normalised to the broadband flux response in the same block,
`R = A_condition / A_FLUX` (`plr_ratio`). The supplementary analysis derives this
outcome and passes it through the same circadian models (M0/M1) as the main analysis.
It is reported in the appendix of the manuscript.

All scripts used only by this analysis start with `si_`. They are run by
`si_pipeline.py` (mirrored in `si_pipeline.ipynb`), which is kept separate from
`pipeline.py`, so running the main pipeline never touches them. No main-pipeline
module is modified by the supplementary analysis.

**Prerequisites:** the per-participant files in `data/retinawise_results/final/`, and
the saliva cosinor (main Step 4). The circadian models use the participant-specific
periods estimated there.

```bash
python si_pipeline.py                     # all steps; Step 5 as dry run, Step 6 skipped
python si_pipeline.py --fit               # all steps, with SMC sampling
python si_pipeline.py --steps 3 3b        # derive the outcome and draw the endpoint figures
python si_pipeline.py --steps 5 5b --fit --outcomes log_plr_ratio
```

Step IDs follow `pipeline.py`, so the same number means the same stage in both.

| Step | Module | What it does | Output |
|------|--------|--------------|--------|
| 3 | `si_derive_protocol_plr` | Takes the maximum constriction on every trial, averages within block, divides by FLUX; also computes the reversed order of operations as an audit | `data/derivatives/plr_ratio/` (`block_amplitudes.csv`, `paired_ratios_analysis.csv`, `exclusion_log.csv`, ...), `results/tex/plr_ratio/` |
| 3b | `si_visualise_protocol_plr` | Endpoint, ordering and ratio-distribution figures | `results/figures/plr_ratio/fig_plr_ratio_{endpoint,ordering,ratios}.png` |
| 4b | `si_analyse_plr_ratio` | Distribution check of the outcome (the raw ratio is skewed) | `results/figures/circadian_cosinor/` |
| 5 | `si_analyse_plr_ratio` | M0/M1 SMC fits per non-FLUX condition (FLUX/FLUX = 1 by construction) | `results/tables/circadian_cosinor/<outcome>/`, `data/derivatives/circadian_cosinor/<outcome>/` |
| 5b | `si_analyse_plr_ratio` | Main Step 5b diagnostics, run on the new outcome | `results/{figures,tables}/circadian_cosinor_diagnostics/<outcome>/` |
| 5c | `si_visualise_protocol_plr` | Bayes factors across the three pupil outcomes, plus the main Step 5c fitted-vs-empirical figures | `results/figures/circadian_cosinor/plr_ratio/` |
| 6 | `si_sensitivity_protocol_plr` | Leave-one-participant-out (reported for L-M) and the registered 20-28 h tau range; refits models, so runs only with `--fit` | `results/tables/plr_ratio/sensitivity/` |

`<outcome>` is `plr_ratio` (the registered outcome) or `log_plr_ratio` (the
prespecified log-ratio sensitivity analysis). Output folders are keyed by outcome name
exactly as `amplitude` and `tonic_min` are, and cannot collide with them.

<details>
<summary>CLI options</summary>

| Flag | Default | Description |
|---|---|---|
| `--steps` | all (`3 3b 4b 5 5b 5c 6`) | Steps to run; always executed in pipeline order |
| `--outcomes` | `plr_ratio log_plr_ratio` | Outcomes for Steps 4b-5c (Step 6 always uses `plr_ratio`) |
| `--fit` | -- | Run SMC sampling in Step 5 and the refits in Step 6 (otherwise dry run / skipped) |
| `--force` | -- | Re-sample in Step 5 even if saved InferenceData exists |
| `--smc-draws` | `10000` | Number of SMC draws; do not lower (biases log Z) |
| `--seed` | `1072` | Random seed |

</details>

Tests for the derivation step: `python -m pytest tests`.

## Auxiliary scripts (not part of the pipeline)

These scripts are **not** called by `pipeline.py` and produce **nothing** that is
copied into the manuscript or the reported figures/tables. They are exploratory /
inspection / illustration helpers; any output lands in `results/figures/` for
local viewing only. Run any of them with `python -m civibeplr.<module_name>`.

| Script | Purpose | Output |
|---|---|---|
| `visualise_circadian_pupil_overview.py` | Exploratory 3-panel circadian overview: amplitude histogram, polar amplitude-vs-phase, and example participant trajectories with posterior cosinor fit | `results/figures/circadian_cosinor/` `<condition>/fig_circadian_overview_mm_<condition>.png` |
| `inspect_circadian_model.py` | Pre-fit check of the Step 5 outcome: per-condition Shapiro-Wilk / skewness / kurtosis printed to stdout, plus histogram + Q-Q panels | `results/figures/circadian_cosinor/` `fig_parameter_distribution_<outcome>.png` |
| `plot_protocol_abstract.py` | Graphical-abstract protocol timeline illustrating the chromatic stimulation conditions | `results/figures/abstract_protocol.svg` |

## Pipeline summary

```
RetinaWise raw exports          clean_saliva_data.csv
        │                               │
        ▼                               ▼
  pupil_prep repository        fit_saliva_cosinor_model  
        │                               │
        ▼                               ▼
  aggregate_pupil_traces      saliva_cosinor_best_fit.csv + per-participant saliva τ
        │
        ├────────────────────────────────────────┐
        ▼                                        ▼
  calculcate_descriptive_pupil_metrices    fit_pupil_trace_2step_cosinor_model
        │                                        │
        ▼                                        ▼
  pupil_metrics.csv           pupil_cosinor_2step_param.csv + plots
                                                 │
                                            (optional) 
                                                 │
                                                 ▼
                                    diagnose_pupil_models
                              (tonic scatter, R² distribution)
                                                 │
                                                 │ (+ saliva τ)
                                                 ▼ 
                              analyse_circadian_pupil_parameters
                                                 │
                                                 ▼
                              circadian_cosinor/ (BF₁₀ per condition)
                                                 │
                                             (optional) 
                                                 │
                                                 ▼
                              diagnose_circadian_models
                                                 │
                                                 ▼
                              circadian_cosinor/diagnostics/ + diagnostics_summary.csv
                                                 │
                                          (optional) │
                                                 ▼
                              visualise_circadian_parameter_cosinor
                                                 │
                                                 ▼
                              circadian_cosinor/ fitted_vs_empirical figures
                                                 │
                                                 ▼
                              copy_assets_to_manuscript.sh --all
                                                 │
                                                 ▼
                              manuscript/ (figures/ tables/ tex/)
                                                 │
                                                 ▼
                                     LaTeX compilation
```

## Copying assets to the manuscript folder

After running the pipeline, generated figures, tables, and LaTeX variable snippets must be copied into the manuscript folder before the LaTeX document can be compiled. This is done with the provided shell script:

```bash
# Copy everything (figures, tables, and tex variable snippets)
./copy_assets_to_manuscript.sh --all

# Or copy selectively
./copy_assets_to_manuscript.sh --fig          # figures only
./copy_assets_to_manuscript.sh --tab          # tables only
./copy_assets_to_manuscript.sh --tex          # tex variable snippets only
./copy_assets_to_manuscript.sh --fig --tex    # figures and tex snippets
```

The script copies from `results/` into `manuscript/20260624_civibe_pupillometry/`. Run it from the repository root. Re-run after any pipeline step whose outputs are included in the manuscript. This includes the supplementary-analysis figures, tables and tex snippets from `si_pipeline.py`.

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| Stimulus frequency | 0.4 Hz | Modulation frequency of the chromatic stimulus |
| LP filter cutoff | 0.2 Hz | Separates tonic (< 0.2 Hz) from oscillatory tracking |
| Filter order | 2 | Butterworth order (effective 4th with zero-phase `filtfilt`) |
| Stim window | 0.0 – 5.0 s | Time window treated as stimulus epoch |
| Post-stim buffer | 0 s | Extra seconds appended to cosinor fit window (0 = stimulus window only) |
| Outcome variable | `rel_size_mm` | Baseline-corrected pupil size in mm |
| Saliva period range | 22–28 h | τ scan range for melatonin cosinor |
| Period selection | AIC | Criterion for choosing best τ per participant |

## Possible adaptations

### Analysis window: stimulus-only vs. stimulus + post-stimulus buffer

The stimulus duration is defined by the experimental protocol and encoded in the
raw data (trial phase label "stim", 0.0 to 5.0 s after trial onset). By default
the cosinor fit (Step 3) and AUC metrics (Step 2) use exactly this window. This
is controlled by a single constant in `src/civibeplr/analysis_io.py`:

```python
DEFAULT_ADD_POSTSTIM_SEC = 0  # set to e.g. 0.5 to extend the analysis window by 0.5 s
```

Setting `DEFAULT_ADD_POSTSTIM_SEC` to a positive value appends that many seconds
after the end of the stimulus to the analysis window. This may be appropriate if
there is a known delay of a few hundred milliseconds between stimulus presentation
and the recorded pupil response -- extending the window slightly ensures that the
full driven response is captured before the recovery transient begins.

The extended window affects:

- `fit_pupil_trace_2step_cosinor_model.py` -- cosinor OLS fit on the residual
- `calculcate_descriptive_pupil_metrices.py` -- stim/poststim boundary for mean,
  std, and AUC metrics
- `visualise_pupil_cosinor.py` -- highlighted fit region in decomposition plots

The same value can also be overridden per-run via the `--add-poststim-sec` CLI
flag in Steps 3 and 3c without modifying the source. The value is saved to the
settings JSON (Step 3 `--save-settings-json`) and reloaded automatically when
generating decomposition plots (Step 3c `--load-settings-json`), so fit and
visualisation always use identical windows.
