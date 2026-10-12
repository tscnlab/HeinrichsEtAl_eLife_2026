# CiViBe Pupillometry

Analysis pipeline for pupillary light responses to sinusoidal chromatic stimulation. Data collected during a 40-hour forced desynchrony protocol with an ultra-short sleep-wake rhythm (12 participants).

## Package

- Package name: `civibeplr`, installed via `pip install -e .`
- Conda environment: `civibeplr` (Python 3.11)
- Source: `src/civibeplr/`

## Pipeline (run in order)

| Step | Module | Input | Output |
|------|--------|-------|--------|
| 0 | External repo ([civibe_pupilprep](https://github.com/tscnlab/civibe_pupilprep)) | RetinaWise raw exports | `data/retinawise_results/final/<ID>_final_data.csv` |
| 1 | `aggregate_pupil_traces` | `data/retinawise_results/final/` | `data/aggregated/data.csv` |
| 2 | `calculcate_pupil_metrices` | `data/aggregated/data.csv` | `results/descriptives/pupil_metrics.csv` |
| 3 | `fit_pupil_trace_2step_cosinor_model` | `data/aggregated/data.csv` | `results/tables/pupil_cosinor/pupil_cosinor_2step_param.csv` |
| 3b | `visualise_pupil_cosinor` | Step 3 CSV + `data/aggregated/data.csv` | `results/figures/pupil_cosinor/` |
| 4 | `fit_saliva_cosinor_model` | `data/raw/clean_saliva_data.csv` | `results/tables/saliva_cosinor/saliva_cosinor_best_fit.csv` |
| 5 | `analyse_circadian_pupil_parameters` | Steps 3+4 outputs | `results/tables/circadian_cosinor/` |

Run any step with `python -m civibeplr.<module_name>`. Step 5 requires `--fit` flag to run SMC sampling.

## Key source files

- `data_io.py` — shared data-loading helpers and column name constants; defines default paths
- `analysis_io.py` — stimulus window constants (0–5 s stimulus, post-stim buffer DEFAULT_ADD_POSTSTIM_SEC)
- `utils.py` — number formatting utilities for tables
- `load_final_data.py` — loads per-participant CSVs from `data/retinawise_results/final/`

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Stimulus frequency | 0.4 Hz | Chromatic stimulus modulation frequency |
| LP filter cutoff | 0.2 Hz | Separates tonic from oscillatory component |
| Filter order | 2 | Butterworth (effective 4th-order with `filtfilt`) |
| Stimulus window | 0–5 s | Epoch treated as stimulus |
| Post-stim buffer | 0 s | Appended to cosinor fit window (0 = stimulus window only; set via DEFAULT_ADD_POSTSTIM_SEC in analysis_io.py) |
| Outcome variable | `rel_size_mm` | Baseline-corrected pupil size in mm |
| Saliva period range | 22–28 h | τ scan range for melatonin cosinor |
| Period selection | AIC | Criterion for best τ per participant |

## Chromatic conditions

Melanopsin (Mel), S-cone (S), L–M cone-opponent (L-M), Luminance (LMS), Broadband flux (LMS+Mel)

## Output locations

- Figures: `results/figures/`
- Tables (CSV + LaTeX): `results/tables/`
- Bayesian model traces: `data/derivatives/circadian_cosinor/`

## Dependencies

- Core: numpy, pandas, scipy, statsmodels, matplotlib, seaborn
- Bayesian (Steps 4–5 only): bambi, pymc

## LaTeX variable export

`format_latex_variables(df, row_key_col, fmt=".2f")` in `latex_table_creation.py` converts any flat CSV DataFrame into `\def` lines, one per non-NaN cell:

```
\def\<RowKey><ColName>{value}
```

To add it for a new CSV table:

1. Load the CSV into a DataFrame (or use the one already in memory).
2. Call `format_latex_variables(df, row_key_col="<col that identifies rows>")`.
3. Write the result to a `.tex` file next to the CSV with a filename starting with `vars_`.

Example (saliva results, one row per participant):

```python
from civibeplr.latex_table_creation import format_latex_variables
tex = format_latex_variables(saliva_df, row_key_col="participant_id")
(output_dir / "vars_saliva.tex").write_text(tex, encoding="utf-8")
```

### Naming rules (auto-applied, no manual map needed)

Column names and row keys are converted to CamelCase by splitting on `_` and `-`
and capitalizing each part. Special substitutions applied first:

| Raw token | LaTeX token |
|-----------|-------------|
| `m1_`     | `Mfull`     |
| `m0_`     | `Mnull`     |
| `r2_`     | `Rtwo`      |
| `bf10`    | `Bf`        |
| digits    | kept as-is (unless covered above) |

Examples:
- condition `flux`, column `m1_intercept_estimate` -> `\FluxMfullInterceptEstimate`
- condition `l-m`, column `m1_alpha_lo` -> `\LmMfullAlphaLo`
- condition `lms`, column `log_bf10` -> `\LmsLogBf`
- condition `lms`, column `m0_r2_cond_hi` -> `\LmsMnullRtwoCondHi`

To add a new substitution (e.g. digits in another column name), extend `_to_camel`
in `latex_table_creation.py` with another `.replace()` call before the split.

## Notes

- No test suite exists in this repository.
- Deprecated/exploratory scripts live in `src/civibeplr/not_used/`.
- `data/`, `results/`, and `misc/` are git-ignored — do not assume their contents are present.
- Step 5 (Bayesian SMC) is computationally expensive; `--smc-draws 2000 --seed 1072` are the defaults used in the analysis.


## Rules 

Smart quotes: " "  ' ' (U+201C, U+201D, U+2018, U+2019)
Avois Em/en dashes: — – (U+2014, U+2013)
Avoid unicode characters that will appear as corrupted symbols or replacement characters
Expected Behavior: Claude Code should generate markdown files using only ASCII-safe characters:
Use [x] and [ ] instead of ✅ and ⚠️
Use regular quotes " instead of smart quotes
Use - for bullets and lists instead of •
Use -> instead of → arrows
Use -- instead of em dashes
Use * instead of ⭐ for emphasis