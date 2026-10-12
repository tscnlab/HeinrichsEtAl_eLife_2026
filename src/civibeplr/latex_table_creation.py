from __future__ import annotations

import re as _re
from pathlib import Path

import numpy as np
import pandas as pd

from civibeplr.analysis_io import (
    PARAMETER_TITLE,
    lowercase_first_word,
    DEFAULT_T_STIM_START,
    DEFAULT_T_STIM_END,
    DEFAULT_ADD_POSTSTIM_SEC,
)
from civibeplr.data_io import DEFAULT_CONDITION_COL, DEFAULT_PARTICIPANT_COL
from civibeplr.utils_cosinor import circ_mean, circ_sd

_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_DIR = _PROJECT_DIR / "results"

CONDITION_ORDER = ["flux", "l-m", "lms", "mel", "s"]

MANUSCRIPT_METRICS = [
    ("prestim_mean",   "Baseline pupil size",    "mm"),
    ("stim_mean_pct",  "Mean constriction",       r"\%"),
    ("amplitude_stim", "Peak constriction onset", "mm"),
    ("auc_stim",       "AUC (stimulus)",          r"mm$\cdot$s"),
]


def save_selected_pupil_metrics_latex(
    metrics: pd.DataFrame,
    output_path: Path,
    condition_col: str = DEFAULT_CONDITION_COL,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    label: str = "tab:tab_pupil_descriptive_metrics_selected",
) -> None:
    """Write a manuscript-ready booktabs LaTeX table of key pupil metrics.

    The table shows mean +/- SD across participants (averaged across blocks first)
    for four metrics, one column per chromatic condition.
    """
    metric_keys = [m[0] for m in MANUSCRIPT_METRICS]

    per_pp = (
        metrics
        .groupby([participant_col, condition_col], dropna=False)[metric_keys]
        .mean()
        .reset_index()
    )

    agg = (
        per_pp
        .groupby(condition_col)[metric_keys]
        .agg(["mean", "std"])
    )

    conditions = [c for c in CONDITION_ORDER if c in agg.index]

    metric_headers = [
        rf"\textit{{{label_str}}} [{unit}]"
        for _, label_str, unit in MANUSCRIPT_METRICS
    ]
    col_spec = "l " + " ".join(["r"] * len(MANUSCRIPT_METRICS))

    header_line = (
        r"\textit{Condition} & "
        + " & ".join(metric_headers)
        + r" \\"
    )

    rows = []
    for cond in conditions:
        cond_tex = TEX_CONDITION_SHORT_ACRONYMS.get(cond, cond)
        cells = [cond_tex]
        for col, _, _ in MANUSCRIPT_METRICS:
            m = agg.loc[cond, (col, "mean")]
            s = agg.loc[cond, (col, "std")]
            if np.isnan(m):
                cells.append("--")
            else:
                cells.append(rf"{m:.2f} $\pm$ {s:.2f}")
        rows.append(" & ".join(cells) + r" \\")

    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\caption{{\label{{{label}}}Descriptive pupil response metrics per chromatic"
        r" condition. Values are mean $\pm$ SD across participants"
        r" (averaged over blocks within participant)."
        r" Baseline pupil size: mean absolute pupil diameter during 1 s pre-stimulus window."
        r" Mean constriction: mean percentage change from baseline during stimulus."
        r" Peak constriction onset: minimum baseline-corrected pupil size within"
        r" 1\,s of stimulus onset."
        r" AUC (stimulus): area under the baseline-corrected pupil trace"
        r" over the stimulus window (trapezoidal rule).}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_line,
        r"\midrule",
    ] + rows + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def format_condition_sample_sizes_latex(
    condition_summary_rows: list[dict],
    label: str = "tab:tab_condition_sample_sizes",
) -> str:
    """Build a LaTeX table of per-condition sample sizes.

    Parameters
    ----------
    condition_summary_rows : list[dict]
        List of dicts with keys ``condition``, ``n_observations``, ``n_participants``.
    label : str
        LaTeX cross-reference label.

    Returns
    -------
    str
        Complete LaTeX table string produced by pandas ``to_latex``.
    """
    df = pd.DataFrame(condition_summary_rows)
    df["condition"] = (
        df["condition"]
        .str.replace("-", "", regex=False)
        .apply(lambda c: rf"\ac{{{c}}}")
    )
    df = df.rename(columns={
        "condition": "Model",
        "n_observations": r"$N_{\text{Observations}}$",
        "n_participants": r"$N_{\text{Clusters}}$",
    })
    tex = df.to_latex(
        index=False,
        caption="Per-condition sample sizes for circadian model building.",
        label=label,
        escape=False,
    )
    return tex.replace(r"\begin{table}", r"\begin{table}" + "\n\\centering", 1)

TEX_CONDITION_SHORT_ACRONYMS = {
    "flux": r"\acs{flux}",
    "l-m": r"\acs{lm}",
    "lms": r"\acs{lms}",
    "mel": r"\acs{mel}",
    "s": r"\acs{s}",
}


def format_bf_results_latex(
    results: dict,
    condition: str,
    label: str | None = None,
) -> str:
    """Build a LaTeX booktabs table from the extracted results dict.

    Parameters
    ----------
    results : dict
        Output of ``extract_results()`` for one condition.
    condition : str
        Condition name used in the caption.
    label : str | None
        LaTeX label for cross-referencing.  Defaults to
        ``tab:circadian_pupil_<condition>``.

    Returns
    -------
    str
        Complete LaTeX table environment string.
    """
    if label is None:
        label = f"tab:circadian_pupil_{condition}"

    def _fmt(value) -> str:
        """Format a numeric value for a table cell; NaN → empty."""
        if isinstance(value, float) and np.isnan(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    def _fmt_ci(lo, hi) -> str:
        """Format a CI as [lo, hi]; all-NaN → empty."""
        if (isinstance(lo, float) and np.isnan(lo)) or (isinstance(hi, float) and np.isnan(hi)):
            return ""
        return f"[{lo:.3f}, {hi:.3f}]"

    def _fmt_ci_bold_local(lo, hi) -> str:
        s = _fmt_ci(lo, hi)
        return rf"\textbf{{{s}}}" if s else s

    def _row(cells: list[str]) -> str:
        """Join cells into a LaTeX table row."""
        return " & ".join(cells) + r"\\"

    def _ci_row(
        param_latex: str,
        m1_ci: tuple,
        m0_ci: tuple = (np.nan, np.nan, np.nan),
        indent: bool = False,
    ) -> str:
        """Row with estimate + CI for M1 and optionally M0."""
        est_m1, lo_m1, hi_m1 = m1_ci
        est_m0, lo_m0, hi_m0 = m0_ci
        prefix = r"\hspace{1em}" if indent else ""
        bold_m1 = _ci_excludes_zero(lo_m1, hi_m1)
        bold_m0 = _ci_excludes_zero(lo_m0, hi_m0)
        return _row([
            f"{prefix}{param_latex}",
            _fmt_bold(est_m1) if bold_m1 else _fmt(est_m1),
            _fmt_ci_bold_local(lo_m1, hi_m1) if bold_m1 else _fmt_ci(lo_m1, hi_m1),
            _fmt_bold(est_m0) if bold_m0 else _fmt(est_m0),
            _fmt_ci_bold_local(lo_m0, hi_m0) if bold_m0 else _fmt_ci(lo_m0, hi_m0),
        ])

    def _scalar_row(param_latex: str, value) -> str:
        """Row with a single value in the M1 Estimate column."""
        return _row([
            param_latex,
            _fmt(round(value, 3) if isinstance(value, float) else value),
            "", "", "",
        ])

    fe   = results["fe"]
    re   = results["re_sd"]
    corr = results["re_corr"]

    lines = [
        r"\begin{table}",
        rf"\caption{{\label{{{label}}}Summary of model estimation for model "
        rf"``{condition}''.}}",
        r"\small",
        r"\centering",
        r"\begin{tabular}[t]{lrlrl}",
        r"\toprule",
        r"\textit{Predictor} & \textit{Estimate} & \textit{CI} & "
        r"\textit{Estimate} & \textit{CI}\\",
        r"\midrule",
        # ── Fixed effects ─────────────────────────────────────────────
        r"\addlinespace[0.3em]",
        r"\multicolumn{5}{l}{\textbf{Fixed effects}}\\",
        _ci_row("Intercept", fe["Intercept"], results["fe_m0_intercept"], indent=True),
        _ci_row(r"$\alpha$ (sin)", fe["alpha"], indent=True),
        _ci_row(r"$\beta$ (cos)", fe["beta"], indent=True),
        # ── Random effects ────────────────────────────────────────────
        r"\addlinespace[0.3em]",
        r"\multicolumn{5}{l}{\textbf{Random effects}}\\",
        _ci_row(
            r"SD(Intercept)",
            re["intercept"],
            results["re_sd_m0_intercept"],
            indent=True,
        ),
        _ci_row(r"SD($\alpha$)", re["alpha"], indent=True),
        _ci_row(r"SD($\beta$)", re["beta"], indent=True),
        _ci_row(
            r"$\rho(\text{Intercept},\alpha)$", corr["intercept_alpha"], indent=True
        ),
        _ci_row(r"$\rho(\text{Intercept},\beta)$", corr["intercept_beta"], indent=True),
        _ci_row(r"$\rho(\alpha,\beta)$", corr["alpha_beta"], indent=True),
        # ── Scalars ───────────────────────────────────────────────────
        r"\midrule",
        r"\addlinespace[0.3em]",
        _scalar_row(r"$N_{\text{Observations}}$", results["n_obs"]),
        _scalar_row(r"$N_{\text{Clusters}}$", results["n_clusters"]),
        _scalar_row(r"$\log \mathrm{BF}_{10}$", results["log_bf10"]),
        # ── R² ────────────────────────────────────────────────────────
        r"\addlinespace[0.3em]",
        _ci_row(r"Conditional $R^2$", results["r2_cond_m1"], results["r2_cond_m0"]),
        _ci_row(r"Marginal $R^2$", results["r2_marg_m1"], results["r2_marg_m0"]),
        # ── Derived ───────────────────────────────────────────────────
        r"\addlinespace[0.3em]",
        _scalar_row("Amplitude", results["amplitude"]),
        _scalar_row("Phase", results["phase_rad"]),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def _ci_excludes_zero(lo, hi) -> bool:
    if any(isinstance(v, float) and np.isnan(v) for v in (lo, hi)):
        return False
    return (lo > 0 and hi > 0) or (lo < 0 and hi < 0)


def _fmt(value) -> str:
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_ci(lo, hi) -> str:
    if (isinstance(lo, float) and np.isnan(lo)) or (isinstance(hi, float) and np.isnan(hi)):
        return ""
    return f"[{lo:.3f}; {hi:.3f}]"


def _fmt_bold(value) -> str:
    s = _fmt(value)
    return rf"\textbf{{{s}}}" if s else s


def _fmt_ci_bold(lo, hi) -> str:
    s = _fmt_ci(lo, hi)
    return rf"\textbf{{{s}}}" if s else s


def _row(*cells: str) -> str:
    return " & ".join(cells) + r" \\"


def format_circadian_parameter_estimates_latex(
    all_results: dict,
    parameter_col: str,
    label: str = "tab:tab_circadian_pupil_parameter_estimates",
) -> str:
    """Build a multi-condition parameter estimates table (nested rows by condition).

    Rows: fixed effects + random-effect SDs + conditional R2 + marginal R2,
    grouped by condition with a midrule between groups.
    Columns: Parameter | M1 Estimate | M1 CI | M0 Estimate | M0 CI

    Parameters
    ----------
    all_results : dict
        Mapping condition -> extract_results() dict.
    parameter_col : str
        Outcome variable name used to label the caption (e.g. ``log_amplitude``).
    label : str
        LaTeX cross-reference label.
    """
    conditions = [c for c in CONDITION_ORDER if c in all_results]

    PARAM_ROWS = [
        # (display label, results key path, indent, both_models)
        # Fixed effects
        ("\\textit{Fixed effects}", None, False, False),
        ("Intercept",             ("fe", "Intercept"),         True, True),
        (r"$\alpha$ (sin)",       ("fe", "alpha"),             True, False),
        (r"$\beta$ (cos)",        ("fe", "beta"),              True, False),
        # Random effects
        ("\\textit{Random effects}", None, False, False),
        (r"SD(Intercept)",        ("re_sd", "intercept"),      True, True),
        (r"SD($\alpha$)",         ("re_sd", "alpha"),          True, False),
        (r"SD($\beta$)",          ("re_sd", "beta"),           True, False),
        # R2
        ("\\textit{Model fit}", None, False, False),
        (r"Conditional $R^2$",    ("r2_cond",),                True, True),
        (r"Marginal $R^2$",       ("r2_marg",),                True, False),
        # Cosinor derivatives
        # ("\\textit{Cosinor derivatives}", None, False, False),
        # ("Amplitude",             ("scalar", "amplitude"),      True, False),
        # ("Phase [rad]",           ("scalar", "phase_rad"),      True, False),
    ]

    def get_triple(res, key_path, model):
        """Extract (est, lo, hi) triple from results dict."""
        nan3 = (np.nan, np.nan, np.nan)
        if key_path is None:
            return nan3
        if len(key_path) == 1:
            # R2 keys are r2_cond_m1 / r2_cond_m0 etc.
            key = f"{key_path[0]}_{model}"
            return res.get(key, nan3)
        section, param = key_path
        if section == "scalar":
            return (res.get(param, np.nan), np.nan, np.nan)
        if model == "m1":
            if section == "fe":
                return res["fe"].get(param, nan3)
            if section == "re_sd":
                return res["re_sd"].get(param, nan3)
        else:  # m0
            if section == "fe" and param == "Intercept":
                return res.get("fe_m0_intercept", nan3)
            if section == "re_sd" and param == "intercept":
                return res.get("re_sd_m0_intercept", nan3)
        return nan3

    param_col_tex = PARAMETER_TITLE.get(parameter_col, parameter_col.replace("_", r"\_"))
    lines = [
        r"\begin{table}",
        rf"\caption{{\label{{{label}}}Parameter estimates for circadian cosinor models"
        rf" ({param_col_tex}) across conditions."
        r" Values are posterior means with 95\% credible intervals,"
        r" except $R^2$, reported as posterior median with a 95\% equal-tailed interval."
        r" M1 = cosinor model; M0 = null (intercept-only) model.}",
        r"\small",
        r"\centering",
        r"\begin{tabular}[t]{ll rr rr}",
        r"\toprule",
        r"\textit{Condition} & \textit{Model parameter}"
        r" & \multicolumn{2}{c}{\textit{M1 (cosinor)}}"
        r" & \multicolumn{2}{c}{\textit{M0 (null)}} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r" &  & \textit{Estimate} & \textit{95\% CI}"
        r" & \textit{Estimate} & \textit{95\% CI} \\",
        r"\midrule",
    ]

    for i, cond in enumerate(conditions):
        if i > 0:
            lines.append(r"\midrule")
        res = all_results[cond]
        cond_tex = TEX_CONDITION_SHORT_ACRONYMS.get(cond, cond)
        first_section_header = True

        for label_tex, key_path, indent, both_models in PARAM_ROWS:
            if key_path is None:
                # section header -- condition label on first header row, empty thereafter
                cond_cell = cond_tex if first_section_header else ""
                first_section_header = False
                lines.append(
                    rf"\addlinespace[0.1em]"
                    + "\n"
                    + rf"{cond_cell} & \multicolumn{{5}}{{l}}{{{label_tex}}} \\"
                )
                continue

            m1 = get_triple(res, key_path, "m1")
            m0 = get_triple(res, key_path, "m0") if both_models else (np.nan, np.nan, np.nan)

            prefix = r"\hspace{2em}" if indent else r"\hspace{1em}"

            est_m1, lo_m1, hi_m1 = m1
            est_m0, lo_m0, hi_m0 = m0
            bold_m1 = _ci_excludes_zero(lo_m1, hi_m1)
            bold_m0 = _ci_excludes_zero(lo_m0, hi_m0)
            cond_cell = ""
            lines.append(_row(
                cond_cell,
                f"{prefix}{label_tex}",
                _fmt_bold(est_m1) if bold_m1 else _fmt(est_m1),
                _fmt_ci_bold(lo_m1, hi_m1) if bold_m1 else _fmt_ci(lo_m1, hi_m1),
                _fmt_bold(est_m0) if bold_m0 else _fmt(est_m0),
                _fmt_ci_bold(lo_m0, hi_m0) if bold_m0 else _fmt_ci(lo_m0, hi_m0),
            ))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _to_camel(token: str) -> str:
    """Split a column name or row key on '_' and '-', capitalize each part.

    Special substitutions applied before splitting:
        m1  -> Mfull
        m0  -> Mnull
    """
    token = (
        token
        .replace("m1_", "Mfull_")
        .replace("m0_", "Mnull_")
        .replace("r2_", "Rsquared_")
        .replace("bf10", "Bf")
        .replace("sin2", "sintwo")
        .replace("cos2", "costwo")
        .replace("2f", "twof")
        .replace("%", "perc")
    )
    _digit_words = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    }
    token = _re.sub(r"\d+", lambda m: "".join(_digit_words[d] for d in m.group()), token)
    parts = _re.split(r"[_\-\s]+", token)
    return "".join(part[0].upper() + part[1:] if part else "" for part in parts)


def format_p_value(p: float, decimals: int = 3) -> str:
    r"""Format a p-value for LaTeX without rounding small values to zero.

    Values below ``10**-decimals`` print as ``$<$0.001`` (for 3 decimals);
    all others print with *decimals* decimal places. NaN returns ``"--"``.
    """
    if pd.isna(p):
        return "--"
    threshold = 10 ** -decimals
    if p < threshold:
        return rf"$<${threshold:.{decimals}f}"
    return f"{p:.{decimals}f}"


def format_latex_variables(
    df: pd.DataFrame,
    row_key_col: str,
    fmt: str = ".2f",
    prefix: str = "",
) -> str:
    r"""Convert any flat CSV DataFrame into LaTeX \def variable lines.

    Each non-NaN cell becomes:
        \def\<Prefix><RowKey><ColName>{<value>}

    where <RowKey> and <ColName> are auto-camelcased from the raw strings:
    - tokens are split on '_' and '-', each part capitalized
    - 'm1' -> 'Mfull', 'm0' -> 'Mnull' (applied before splitting)

    Parameters
    ----------
    df : pd.DataFrame
        Flat table with one row per entity (e.g. one row per condition).
    row_key_col : str
        Column whose values become the leading camelcase prefix of each variable.
    fmt : str
        Format string for float values (default ``".2f"``).
    prefix : str
        Optional literal string prepended to every variable name (e.g. ``"Saliva"``).
    """
    lines: list[str] = []
    value_cols = [c for c in df.columns if c != row_key_col]
    for _, row in df.iterrows():
        raw_key = str(row[row_key_col])
        row_prefix = _to_camel(raw_key)
        for col in value_cols:
            val = row[col]
            if isinstance(val, float) and np.isnan(val):
                continue
            col_suffix = _to_camel(col)
            if isinstance(val, float):
                formatted = format(val, fmt)
            else:
                formatted = str(val)
            lines.append(rf"\def\{_to_camel(prefix) if prefix else ''}{row_prefix}{col_suffix}{{{formatted}}}")
    return "\n".join(lines) + "\n"


_CIRCADIAN_OUTCOME_UNIT: dict[str, str] = {
    "amplitude":       "mm",
    "amplitude_2f":    "mm",
    "tonic_min":       "mm",
    "tonic_mean":      "mm",
    "baseline":        "mm",
    "phase":           "rad",
    "phase_2f":        "rad",
    "log_amplitude":   "log mm",
    "log_amplitude_2f": "log mm",
}


def format_circadian_model_summary_latex(
    all_results: dict,
    parameter_col: str,
    label: str = "tab:tab_circadian_model_summary",
) -> str:
    """Build a one-row-per-condition model summary table.

    For log-scale outcomes (parameter_col starts with "log"), an extra column
    is added: Amplitude (mm), the back-transformed circadian modulation depth in
    original units, computed as exp(intercept) * sinh(A_log) with 95% CI.

    Parameters
    ----------
    all_results : dict
        Mapping condition -> extract_results() dict.
    parameter_col : str
        Outcome variable name used to label the caption.
    label : str
        LaTeX cross-reference label.
    """
    conditions = [c for c in CONDITION_ORDER if c in all_results]
    is_log = parameter_col.startswith("log")

    param_col_tex = PARAMETER_TITLE.get(parameter_col, parameter_col.replace("_", r"\_"))

    amp_mm_note = (
        r" Amplitude [mm] = $\exp(\hat{\mu})\,\sinh(\hat{A}_{\log})$:"
        r" half the peak-to-trough range of the circadian rhythm"
        r" back-transformed to original units (95\% CI in brackets)."
        if is_log else ""
    )

    unit = _CIRCADIAN_OUTCOME_UNIT.get(parameter_col)
    unit_str = f" [{unit}]" if unit else ""
    # Short noun phrase for the outcome, used inside the caption sentence; avoids
    # the awkward "modulation of the amplitude (fundamental)" doubling.
    outcome_phrase = {
        "amplitude": "phasic fundamental amplitude",
        "amplitude_2f": "phasic second-harmonic amplitude",
        "log_amplitude": "log phasic fundamental amplitude",
        "log_amplitude_2f": "log phasic second-harmonic amplitude",
        "tonic_min": "tonic minimum",
        "tonic_mean": "tonic mean",
        "baseline": "baseline pupil size",
        "phase": "fundamental phase",
        "phase_2f": "second-harmonic phase",
    }.get(parameter_col, lowercase_first_word(param_col_tex))
    estimates_label = f"tab:tab_circadian_parameter_estimates_{parameter_col}"
    amp_phase_note = (
        rf" $A_{{\mathrm{{circ}}}}${unit_str} and $\phi_{{\mathrm{{circ}}}}$"
        r" [rad] are the amplitude and phase of the fitted circadian"
        rf" modulation of the {outcome_phrase}, derived from the population-level"
        r" posterior sine and cosine slopes and reported as posterior mean (SD);"
        r" for the phase, the circular mean and circular SD of the posterior draws."
        r" This SD is group-level estimation uncertainty only; across-participant"
        r" variability is given by the random-effect SDs in"
        rf" Table~\ref{{{estimates_label}}}."
    )

    col_spec  = r"lrrrrrr" if is_log else r"lrrrrr"
    amp_col   = r" & \textit{Amplitude [mm]}" if is_log else ""

    lines = [
        r"\begin{table}",
        rf"\caption{{\label{{{label}}}Model comparison and population-level circadian"
        rf" estimates for the {outcome_phrase}, per chromatic condition."
        r" $N_{\text{obs}}$, retained block-level observations;"
        r" $N_{\text{part}}$, participants;"
        r" \logbf{}, log Bayes factor (base 10) comparing the circadian model M1"
        r" with the constant-level model M0."
        + amp_phase_note
        + amp_mm_note
        + "}",
        r"\centering",
        rf"\begin{{tabular}}[t]{{{col_spec}}}",
        r"\toprule",
        r"\textit{Condition}"
        r" & $N_{\text{obs}}$"
        r" & $N_{\text{part}}$"
        r" & \logbf"
        + rf" & $A_{{\mathrm{{circ}}}}${unit_str}"
        + amp_col
        + r" & $\phi_{\mathrm{circ}}$ [rad]\\",
        r"\midrule",
    ]

    for cond in conditions:
        res = all_results[cond]
        cond_tex = TEX_CONDITION_SHORT_ACRONYMS.get(cond, cond)
        def _fmt_ci(med, lo, hi, decimals=3) -> str:
            """Format median with 95% CI."""
            fmt = f"{{:.{decimals}f}}"
            return f"{fmt.format(med)} [{fmt.format(lo)}; {fmt.format(hi)}]"

        amp_ci = res.get("amplitude_ci") or (float("nan"), float("nan"))
        amp_lo, amp_hi = amp_ci[0], amp_ci[1]
        amp_sd = res.get("amplitude_sd", float("nan"))
        pha_ci = res.get("phase_ci") or (float("nan"), float("nan"))
        pha_lo, pha_hi = pha_ci[0], pha_ci[1]
        pha_sd = res.get("phase_sd", float("nan"))

        def _fmt_mean_sd_ci(mean, sd, lo, hi, decimals=3) -> str:
            fmt = f"{{:.{decimals}f}}"
            # return f"{fmt.format(mean)} ({fmt.format(sd)}) [{fmt.format(lo)}; {fmt.format(hi)}]"
            return f"{fmt.format(mean)} ({fmt.format(sd)})"

        row_cells = [
            cond_tex,
            str(res["n_obs"]),
            str(res["n_clusters"]),
            _fmt(res["log_bf10"]),
            _fmt_mean_sd_ci(res["amplitude"], amp_sd, amp_lo, amp_hi),
        ]
        if is_log:
            amp_mm = res.get("amplitude_mm")
            if amp_mm is not None and not (isinstance(amp_mm[0], float) and np.isnan(amp_mm[0])):
                med, lo, hi = amp_mm
                row_cells.append(f"{med:.3f} [{lo:.3f}; {hi:.3f}]")
            else:
                row_cells.append("")
        row_cells.append(_fmt_mean_sd_ci(res["phase_rad"], pha_sd, pha_lo, pha_hi))
        lines.append(_row(*row_cells))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


PARAMETER_DISPLAY_NAMES: dict[str, str] = {
    "amplitude":    r"Amplitude$_f$ [mm]",
    "amplitude_2f": r"Amplitude$_{2f}$ [mm]",
    "phase":        r"Phase$_f$ [rad]",
    "phase_2f":     r"Phase$_{2f}$ [rad]",
    "tonic_min":    r"Tonic minimum [mm]",
    "baseline":     r"Baseline [mm]",
}


def format_circadian_model_summary_all_metrics_latex(
    all_metrics_results: dict[str, dict[str, dict]],
    label: str = "tab:tab_circadian_model_summary_all_metrics",
) -> str:
    """Build a combined model summary table spanning all pupil metrics.

    Parameters
    ----------
    all_metrics_results : dict
        Mapping metric_name -> {condition -> extract_results() dict}.
        Each inner dict is the ``all_results`` produced by one ``main()`` run.
    label : str
        LaTeX cross-reference label.

    Returns
    -------
    str
        Complete LaTeX table environment string.

    Table layout
    ------------
    Rows grouped by metric (with a \\midrule between groups); within each
    group one row per chromatic condition.
    Columns: Condition | N_obs | N_clust | log BF10 | Amplitude (SD) | Phase (SD)
    """
    metrics = list(all_metrics_results.keys())
    conditions = [c for c in CONDITION_ORDER if any(c in all_metrics_results[m] for m in metrics)]

    def _fmt_mean_sd(mean, sd, decimals=3) -> str:
        if isinstance(mean, float) and np.isnan(mean):
            return "--"
        fmt = f"{{:.{decimals}f}}"
        return f"{fmt.format(mean)} ({fmt.format(sd)})"

    lines = [
        r"\begin{table*}",
        rf"\caption{{\label{{{label}}}Model summary for circadian cosinor fits"
        r" across all pupil metrics and chromatic conditions."
        r" $N_{\text{obs}}$ = number of observations;"
        r" $N_{\text{part}}$ = number of participants;"
        r" $\log \mathrm{BF}_{10}$ = log Bayes factor (base 10, SMC marginal likelihood ratio)."
        r" Circadian amplitude and phase are derived from population-level posterior"
        r" sin/cos slopes; reported as posterior mean (SD), with the circular mean and"
        r" circular SD for the phase.}",
        r"\centering",
        r"\small",
        r"\begin{tabular}[t]{llrrrll}",
        r"\toprule",
        r"\textit{Metric}"
        r" & \textit{Condition}"
        r" & $N_{\text{obs}}$"
        r" & $N_{\text{part}}$"
        r" & $\log \mathrm{BF}_{10}$"
        r" & $A_{\mathrm{circ}}$ (SD)"
        r" & $\phi_{\mathrm{circ}}$ (SD) \\",
        r"\midrule",
    ]

    for i, metric in enumerate(metrics):
        if i > 0:
            lines.append(r"\midrule")
        metric_label = PARAMETER_DISPLAY_NAMES.get(metric, metric.replace("_", r"\_"))
        metric_conditions = [c for c in CONDITION_ORDER if c in all_metrics_results[metric]]
        for j, cond in enumerate(metric_conditions):
            res = all_metrics_results[metric][cond]
            metric_cell = metric_label if j == 0 else ""
            cond_tex = TEX_CONDITION_SHORT_ACRONYMS.get(cond, cond)
            amp_sd  = res.get("amplitude_sd", float("nan"))
            pha_sd  = res.get("phase_sd", float("nan"))
            row_cells = [
                metric_cell,
                cond_tex,
                str(res["n_obs"]),
                str(res["n_clusters"]),
                _fmt(res["log_bf10"]),
                _fmt_mean_sd(res["amplitude"], amp_sd),
                _fmt_mean_sd(res["phase_rad"], pha_sd),
            ]
            lines.append(_row(*row_cells))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


_COSINOR_2STEP_SCALAR_COLS = [
    "tonic_mean", "tonic_min", "tonic_min_time",
    "amplitude", "phase_rad",
    "alpha_sin", "beta_cos",
    "alpha_sin_ci_lo", "alpha_sin_ci_hi",
    "beta_cos_ci_lo", "beta_cos_ci_hi",
    "amplitude_2f", "phase_2f_rad",
    "alpha_sin_2f", "beta_cos_2f",
    "alpha_sin_2f_ci_lo", "alpha_sin_2f_ci_hi",
    "beta_cos_2f_ci_lo", "beta_cos_2f_ci_hi",
    "amplitude_ratio",
    "adj_r_squared",
]


# Phase angles are averaged with circular statistics at every aggregation step.
_COSINOR_2STEP_PHASE_COLS = {"phase_rad", "phase_2f_rad"}


def _summarise_cols(data, cols: list[str], how: str):
    """Mean or SD of ``cols`` for a DataFrame or GroupBy, circular for phase columns."""
    circ = {"mean": circ_mean, "std": circ_sd}[how]
    out = data[cols].agg(how)
    for col in cols:
        if col in _COSINOR_2STEP_PHASE_COLS:
            # Series.agg would apply a plain callable element-wise, so call it directly
            out[col] = circ(data[col]) if isinstance(data, pd.DataFrame) else data[col].agg(circ)
    return out


def aggregate_pupil_cosinor_2step(
    results_df: pd.DataFrame,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate raw 2-step cosinor results across blocks and participants.

    Phase columns use the circular mean and circular SD.

    Returns
    -------
    per_pp : DataFrame
        Mean per participant x condition (blocks averaged out).
    agg : DataFrame
        MultiIndex DataFrame with (col, mean/std) per condition.
    across_cond : DataFrame
        Mean per participant across conditions (for grand mean/SD).
    """
    cols = _COSINOR_2STEP_SCALAR_COLS
    per_pp = _summarise_cols(
        results_df.groupby([participant_col, condition_col], dropna=False), cols, "mean"
    ).reset_index()
    by_cond = per_pp.groupby(condition_col)
    agg = pd.concat(
        {"mean": _summarise_cols(by_cond, cols, "mean"),
         "std": _summarise_cols(by_cond, cols, "std")},
        axis=1,
    ).swaplevel(axis=1)
    across_cond = _summarise_cols(per_pp.groupby(participant_col), cols, "mean")
    return per_pp, agg, across_cond


def save_pupil_cosinor_2step_tonic_latex(
    agg: pd.DataFrame,
    across_cond: pd.DataFrame,
    output_dir: Path | None = None,
    label: str = "tab:tab_pupil_cosinor_2step_tonic",
) -> None:
    """Write the tonic-component LaTeX table for the 2-step cosinor results.

    Parameters
    ----------
    agg, across_cond :
        Pre-aggregated DataFrames from ``aggregate_pupil_cosinor_2step``.
    output_dir : Path | None
        Root results directory. Defaults to ``DEFAULT_RESULT_DIR``.
    """
    if output_dir is None:
        output_dir = DEFAULT_RESULT_DIR

    grand_mean = _summarise_cols(across_cond, list(across_cond.columns), "mean")
    grand_std  = _summarise_cols(across_cond, list(across_cond.columns), "std")
    conditions = [c for c in CONDITION_ORDER if c in agg.index]

    def ms(col: str, cond: str) -> str:
        m = agg.loc[cond, (col, "mean")]
        s = agg.loc[cond, (col, "std")]
        if np.isnan(m):
            return "--"
        return rf"{m:.2f} $\pm$ {s:.2f}"

    def ms_grand(col: str) -> str:
        m = grand_mean[col]
        s = grand_std[col]
        if np.isnan(m):
            return "--"
        return rf"{m:.2f} $\pm$ {s:.2f}"

    lines = [
        r"\begin{table}",
        r"\centering",
        #r"\scriptsize",
        rf"\caption{{\label{{{label}}}Tonic pupil component per chromatic condition."
        r" Values are mean $\pm$ SD across participants, with blocks averaged within"
        r" participant; the $M \pm SD$ row averages these participant values across"
        r" conditions. The tonic minimum is the most negative deflection of the"
        r" low-pass filtered component within the 5-s stimulus window, and its latency"
        r" is measured from stimulus onset.}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textit{Condition} & \textit{Tonic mean [mm]} & \textit{Tonic minimum [mm]} & \textit{Tonic minimum latency [s]} \\",
        r"\midrule",
    ]
    for cond in conditions:
        cond_tex = TEX_CONDITION_SHORT_ACRONYMS.get(cond, cond)
        cells = [cond_tex, ms("tonic_mean", cond), ms("tonic_min", cond), ms("tonic_min_time", cond)]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join([r"$M \pm SD$", ms_grand("tonic_mean"), ms_grand("tonic_min"), ms_grand("tonic_min_time")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out = output_dir / "tables" / "pupil_cosinor" / "tab_pupil_cosinor_2step_tonic.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX table: {out}")


def save_pupil_cosinor_2step_phasic_latex(
    agg: pd.DataFrame,
    across_cond: pd.DataFrame,
    output_dir: Path | None = None,
    label: str = "tab:tab_pupil_cosinor_2step_phasic",
    show_ci: bool = False,
) -> None:
    """Write the phasic-component LaTeX table for the 2-step cosinor results.

    Parameters
    ----------
    agg, across_cond :
        Pre-aggregated DataFrames from ``aggregate_pupil_cosinor_2step``.
    output_dir : Path | None
        Root results directory. Defaults to ``DEFAULT_RESULT_DIR``.
    show_ci : bool
        If True, include 95% CI columns for both alpha_sin and beta_cos.
        If False, omit all CI columns.
    """
    if output_dir is None:
        output_dir = DEFAULT_RESULT_DIR

    grand_mean = _summarise_cols(across_cond, list(across_cond.columns), "mean")
    grand_std  = _summarise_cols(across_cond, list(across_cond.columns), "std")
    conditions = [c for c in CONDITION_ORDER if c in agg.index]

    def ms(col: str, cond: str) -> str:
        m = agg.loc[cond, (col, "mean")]
        s = agg.loc[cond, (col, "std")]
        if np.isnan(m):
            return "--"
        return rf"{m:.2f} $\pm$ {s:.2f}"

    def ci_cell(lo_col: str, hi_col: str, cond: str) -> str:
        lo = agg.loc[cond, (lo_col, "mean")]
        hi = agg.loc[cond, (hi_col, "mean")]
        if np.isnan(lo) or np.isnan(hi):
            return "--"
        return rf"[{lo:.2f}; {hi:.2f}]"

    def ms_grand(col: str) -> str:
        m = grand_mean[col]
        s = grand_std[col]
        if np.isnan(m):
            return "--"
        return rf"{m:.2f} $\pm$ {s:.2f}"

    def ci_grand(lo_col: str, hi_col: str) -> str:
        lo = grand_mean[lo_col]
        hi = grand_mean[hi_col]
        if np.isnan(lo) or np.isnan(hi):
            return "--"
        return rf"[{lo:.2f}; {hi:.2f}]"

    ci_caption = (
        r" 95\,\% CIs in brackets (mean across participants)." if show_ci else ""
    )
    if show_ci:
        col_spec = r"\begin{tabular}{lrrrrrrrrrrrrrrr}"
        header_span = (
            r" & \multicolumn{6}{c}{\textit{Fundamental}}"
            r" & \multicolumn{6}{c}{\textit{2nd harmonic}}"
        )
        cmidrule = r"\cmidrule(lr){2-7}\cmidrule(lr){8-13}\cmidrule(lr){14-15}"
        header_cols = (
            r" & $A_f$ [mm] & $\phi_f$ [rad]"
            r" & $\alpha_f$ & \textit{95\,\% CI}"
            r" & $\beta_f$ & \textit{95\,\% CI}"
            r" & $A_{2f}$ [mm] & $\phi_{2f}$ [rad]"
            r" & $\alpha_{2f}$ & \textit{95\,\% CI}"
            r" & $\beta_{2f}$ & \textit{95\,\% CI}"
            r" & $A_f/A_{2f}$ & adj. $R^2$ \\"
        )
    else:
        col_spec = r"\begin{tabular}{lrrrrrrrrrrr}"
        header_span = (
            r" & \multicolumn{4}{c}{\textit{Fundamental}}"
            r" & \multicolumn{4}{c}{\textit{2nd harmonic}}"
        )
        cmidrule = r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-10}\cmidrule(lr){11-11}"
        header_cols = (
            r" & $A_f$ [mm] & $\phi_f$ [rad]"
            r" & $\alpha_f$ & $\beta_f$"
            r" & $A_{2f}$ [mm] & $\phi_{2f}$ [rad]"
            r" & $\alpha_{2f}$ & $\beta_{2f}$"
            r" & $A_f/A_{2f}$ & adj. $R^2$ \\"
        )
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{\label{{{label}}}Phasic pupil component per chromatic condition."
        r" Values are mean $\pm$ SD across participants, with blocks averaged within"
        r" participant; the $M \pm SD$ row averages these participant values across"
        r" conditions. $A_f/A_{2f}$ is computed for each block and then averaged in"
        r" the same way, so it is not the ratio of the displayed $A_f$ and $A_{2f}$"
        r" means. The fundamental ($f$) and second-harmonic ($2f$) components"
        r" correspond to 0.4 and 0.8\,Hz. Phases $\phi$ are summarised by the circular"
        r" mean and circular SD at every averaging step."
        r" $\alpha$ and $\beta$ are the sine and cosine"
        rf" slopes of the cosinor fit.{ci_caption} Adjusted $R^2$ refers to the"
        r" bi-harmonic cosinor fitted to the residual after subtraction of the tonic"
        r" component.}",
        col_spec,
        r"\toprule",
        header_span + r" & \multicolumn{1}{c}{\textit{Ratio}} & \multicolumn{1}{c}{\textit{Fit}} \\",
        cmidrule,
        r"\textit{Condition}" + header_cols,
        r"\midrule",
    ]
    for cond in conditions:
        cond_tex = TEX_CONDITION_SHORT_ACRONYMS.get(cond, cond)
        cells = [
            cond_tex,
            ms("amplitude", cond),
            ms("phase_rad", cond),
            ms("alpha_sin", cond),
        ]
        if show_ci:
            cells.append(ci_cell("alpha_sin_ci_lo", "alpha_sin_ci_hi", cond))
        cells.append(ms("beta_cos", cond))
        if show_ci:
            cells.append(ci_cell("beta_cos_ci_lo", "beta_cos_ci_hi", cond))
        cells += [
            ms("amplitude_2f", cond),
            ms("phase_2f_rad", cond),
            ms("alpha_sin_2f", cond),
        ]
        if show_ci:
            cells.append(ci_cell("alpha_sin_2f_ci_lo", "alpha_sin_2f_ci_hi", cond))
        cells.append(ms("beta_cos_2f", cond))
        if show_ci:
            cells.append(ci_cell("beta_cos_2f_ci_lo", "beta_cos_2f_ci_hi", cond))
        cells.append(ms("amplitude_ratio", cond))
        cells.append(ms("adj_r_squared", cond))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    grand_cells = [
        r"$M \pm SD$",
        ms_grand("amplitude"),
        ms_grand("phase_rad"),
        ms_grand("alpha_sin"),
    ]
    if show_ci:
        grand_cells.append(ci_grand("alpha_sin_ci_lo", "alpha_sin_ci_hi"))
    grand_cells.append(ms_grand("beta_cos"))
    if show_ci:
        grand_cells.append(ci_grand("beta_cos_ci_lo", "beta_cos_ci_hi"))
    grand_cells += [
        ms_grand("amplitude_2f"),
        ms_grand("phase_2f_rad"),
        ms_grand("alpha_sin_2f"),
    ]
    if show_ci:
        grand_cells.append(ci_grand("alpha_sin_2f_ci_lo", "alpha_sin_2f_ci_hi"))
    grand_cells.append(ms_grand("beta_cos_2f"))
    if show_ci:
        grand_cells.append(ci_grand("beta_cos_2f_ci_lo", "beta_cos_2f_ci_hi"))
    grand_cells.append(ms_grand("amplitude_ratio"))
    grand_cells.append(ms_grand("adj_r_squared"))
    lines.append(" & ".join(grand_cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    out = output_dir / "tables" / "pupil_cosinor" / "tab_pupil_cosinor_2step_phasic.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX table: {out}")


def save_pupil_cosinor_2step_latex(
    results_df: pd.DataFrame,
    output_dir: Path | None = None,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
) -> None:
    """Write tonic/phasic LaTeX tables and summary ``\\def`` variables for 2-step cosinor results.

    The function first aggregates results with :func:`aggregate_pupil_cosinor_2step`,
    then exports two tables:
    1) tonic parameters and 2) phasic parameters.

    It also exports LaTeX variables for selected metrics
    (``tonic_mean``, ``tonic_min``, ``tonic_min_time``, ``amplitude``, ``phase_rad``,
    ``alpha_sin``, ``beta_cos``, ``amplitude_2f``, ``amplitude_ratio``, ``adj_r_squared``).

    Average calculation:
    - Per-condition values are the mean of participant-level rows within each
        condition.
    - Grand summary values are the mean and SD across participants of
        ``across_cond`` (one condition-averaged value per participant).
    - Phase columns use the circular mean and circular SD at every step.
    - These are written as ``\def\PupilCosinorMean...`` and
        ``\def\PupilCosinorStd...`` entries, rounded to two decimals.
    """
    if output_dir is None:
        output_dir = DEFAULT_RESULT_DIR

    per_pp, agg, across_cond = aggregate_pupil_cosinor_2step(
        results_df, participant_col=participant_col, condition_col=condition_col
    )

    save_pupil_cosinor_2step_tonic_latex(agg, across_cond, output_dir=output_dir)
    save_pupil_cosinor_2step_phasic_latex(agg, across_cond, output_dir=output_dir)
    save_pupil_cosinor_2step_by_participant(per_pp, output_dir=output_dir,
                                            participant_col=participant_col,
                                            condition_col=condition_col)

    # ── format_latex_variables export ─────────────────────────────────────
    vars_cols = ["tonic_mean", "tonic_min", "tonic_min_time",
                 "amplitude", "phase_rad", "alpha_sin", "beta_cos",
                 "amplitude_2f", "amplitude_ratio", "adj_r_squared"] #"r_squared" -> not available
    
    # Per-condition means for the variables of interest, exported as LaTeX variables.
    summary_df = _summarise_cols(per_pp.groupby(condition_col), vars_cols, "mean").reset_index()
    # Per-condition SDs, exported as \def\PupilCosinor<Cond><Var>Sd so the text can
    # cite the same dispersion shown in the tables (means come from summary_df below).
    sd_df = _summarise_cols(per_pp.groupby(condition_col), vars_cols, "std").reset_index()
    # Pre-camelcase condition values so format_latex_variables uses _to_camel
    # instead of \acs{...} macros (which are illegal inside \def names).
    summary_df[condition_col] = summary_df[condition_col].map(_to_camel)
    tex_vars = format_latex_variables(summary_df, row_key_col=condition_col, prefix="Pupil Cosinor")

    sd_lines = []
    for _, row in sd_df.iterrows():
        camel_cond = _to_camel(row[condition_col])
        for col in vars_cols:
            s = row[col]
            if not pd.isna(s):
                sd_lines.append(
                    rf"\def\PupilCosinor{camel_cond}{_to_camel(col)}Sd{{{s:.2f}}}"
                )

    # Grand mean/SD and the min/max across participants (across_cond holds one
    # condition-averaged value per participant, so Min/Max is the between-participant
    # range that brackets the same SD -- not the block-level range).
    grand_mean_row = _summarise_cols(across_cond, vars_cols, "mean")
    grand_std_row  = _summarise_cols(across_cond, vars_cols, "std")
    grand_min_row  = across_cond[vars_cols].min()
    grand_max_row  = across_cond[vars_cols].max()
    grand_lines = []
    for col in vars_cols:
        camel_col = _to_camel(col)
        m = grand_mean_row[col]
        s = grand_std_row[col]
        lo = grand_min_row[col]
        hi = grand_max_row[col]
        if not np.isnan(m):
            grand_lines.append(rf"\def\PupilCosinorMean{camel_col}{{{m:.2f}}}")
        if not np.isnan(s):
            grand_lines.append(rf"\def\PupilCosinorStd{camel_col}{{{s:.2f}}}")
        if not np.isnan(lo):
            grand_lines.append(rf"\def\PupilCosinorMin{camel_col}{{{lo:.2f}}}")
        if not np.isnan(hi):
            grand_lines.append(rf"\def\PupilCosinorMax{camel_col}{{{hi:.2f}}}")

    # Block-level edge diagnostics for the tonic-minimum latency: individual blocks
    # whose within-window minimum fell exactly on the [start, end] detection boundary.
    # Computed from the raw block rows (not participant-averaged), so it exposes the
    # noise that participant-level aggregation removes.
    if "tonic_min_time" in results_df.columns:
        t = results_df["tonic_min_time"].dropna()
        win_lo = DEFAULT_T_STIM_START
        win_hi = DEFAULT_T_STIM_END + DEFAULT_ADD_POSTSTIM_SEC
        tol = 1e-6
        edge_mask = (results_df["tonic_min_time"] <= win_lo + tol) | (
            results_df["tonic_min_time"] >= win_hi - tol
        )
        n_blocks = int(t.shape[0])
        n_edge = int(edge_mask.sum())
        n_edge_pp = int(results_df.loc[edge_mask, participant_col].nunique())
        grand_lines.append(rf"\def\PupilCosinorTonicMinTimeNBlocks{{{n_blocks}}}")
        grand_lines.append(rf"\def\PupilCosinorTonicMinTimeEdgeBlocks{{{n_edge}}}")
        grand_lines.append(rf"\def\PupilCosinorTonicMinTimeEdgeParticipants{{{n_edge_pp}}}")
        # Per-condition edge counts, so the text can show that boundary placements
        # are not evenly distributed across conditions.
        for cond, sub in results_df.groupby(condition_col, dropna=False):
            camel_cond = _to_camel(str(cond))
            sub_t = sub["tonic_min_time"].dropna()
            sub_edge = (sub["tonic_min_time"] <= win_lo + tol) | (
                sub["tonic_min_time"] >= win_hi - tol
            )
            grand_lines.append(
                rf"\def\PupilCosinor{camel_cond}TonicMinTimeBlocks{{{int(sub_t.shape[0])}}}"
            )
            grand_lines.append(
                rf"\def\PupilCosinor{camel_cond}TonicMinTimeEdgeBlocks{{{int(sub_edge.sum())}}}"
            )

    out_vars = output_dir / "tex" / "pupil_cosinor" / "vars_pupil_cosinor_2step.tex"
    out_vars.parent.mkdir(parents=True, exist_ok=True)
    out_vars.write_text(
        tex_vars + "\n" + "\n".join(sd_lines) + "\n\n" + "\n".join(grand_lines) + "\n",
        encoding="utf-8",
    )
    print(f"Saved LaTeX variables: {out_vars}")


def save_pupil_cosinor_2step_by_participant(
    per_pp: pd.DataFrame,
    output_dir: Path | None = None,
    participant_col: str = DEFAULT_PARTICIPANT_COL,
    condition_col: str = DEFAULT_CONDITION_COL,
) -> None:
    """Save participant x condition averaged 2-step cosinor results as CSV and LaTeX vars.

    The CSV contains one row per participant x condition (blocks averaged within
    each participant).  The vars .tex file exports \def macros with the naming
    scheme ``\PpCosinor<ParticipantId><Condition><Column>`` so they can be
    referenced in hand-written LaTeX documents.

    Parameters
    ----------
    per_pp :
        Participant x condition DataFrame from ``aggregate_pupil_cosinor_2step``
        (blocks already averaged within each participant x condition cell).
    output_dir :
        Root results directory. Defaults to ``DEFAULT_RESULT_DIR``.
    """
    if output_dir is None:
        output_dir = DEFAULT_RESULT_DIR

    csv_cols = [participant_col, condition_col] + [c for c in _COSINOR_2STEP_SCALAR_COLS if c in per_pp.columns]
    csv_out = output_dir / "tables" / "pupil_cosinor" / "pupil_cosinor_2step_by_participant.csv"
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    per_pp[csv_cols].to_csv(csv_out, index=False)
    print(f"Saved CSV: {csv_out}")

    vars_cols = ["amplitude", "amplitude_2f", "amplitude_ratio",
                 "tonic_mean", "tonic_min", "tonic_min_time", "adj_r_squared"]
    available = [c for c in vars_cols if c in per_pp.columns]

    export = per_pp[[participant_col, condition_col] + available].copy()
    export["_row_key"] = (
        export[participant_col].astype(str).str.replace(r"\s+", "", regex=True)
        + "_"
        + export[condition_col].astype(str)
    )
    tex = format_latex_variables(export.drop(columns=[participant_col, condition_col]),
                                 row_key_col="_row_key", fmt=".4f",
                                 prefix="PpCosinor")

    tex_out = output_dir / "tex" / "pupil_cosinor" / "vars_pupil_cosinor_2step_by_participant.tex"
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.write_text(tex, encoding="utf-8")
    print(f"Saved LaTeX variables: {tex_out}")


def format_vars_pupil_cosinor_to_latex(results_df: pd.DataFrame, file_path: Path =  DEFAULT_RESULT_DIR / "tex" / "pupil_cosinor_diagnostics" / "vars_pupil_cosinor_summary.tex") -> None:
    """Compute per-condition means and write LaTeX \\def variables to a .tex file.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing the results.
    file_path : Path, optional
        Path to the output .tex file (default: DEFAULT_RESULT_DIR / "tex" / "pupil_cosinor_diagnostics" / "vars_pupil_cosinor_summary.tex").
    """
    summary_cols = ["r_squared", "adj_r_squared"]
    available = [c for c in summary_cols if c in results_df.columns]
    summary_df = (
        results_df
        .groupby(DEFAULT_CONDITION_COL, dropna=False)[available]
        .mean()
        .reset_index()
        .rename(columns={DEFAULT_CONDITION_COL: "condition"})
    )

    tex = format_latex_variables(summary_df, row_key_col="condition")
    file_path.write_text(tex, encoding="utf-8")
    print(f"Saved LaTeX variables to: {file_path}")
