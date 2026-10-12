"""Extract and export participant demographic and questionnaire information.

Sources:
  - Lab screening (REDCap): anthropometrics, sex, handedness
  - Online screening (REDCap): gender, health, smoking, ESS, AUDIT, SRQ, PSQI, MCTq
  - Session info (Excel): sex fallback for missing values; startdate_dl for age calculation

Outputs saved to results/tables/participant_info/:
  - participant_demographics.csv
  - participant_demographics.tex
  - vars_participant_demographics.tex
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from civibeplr.latex_table_creation import format_latex_variables

_PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = _PROJECT_DIR / "results"

LAB_SCREENING_PATH = (
    _PROJECT_DIR
    / "data/raw/screening/group/lab-screening/Civibelabscreening_DATA_2024-11-26_1119.csv"
)
ONLINE_SCREENING_PATH = (
    _PROJECT_DIR
    / "data/raw/screening/group/online-screening/Civibeonlinescreenin_DATA_2025-08-20_1049.csv"
)
SESSION_INFO_PATH = (
    _PROJECT_DIR / "data/raw/screening/group/20260423_session-info.xlsx"
)

# selected participants - based on lateron selection (from REDCap and session info)
PARTICIPANTS = [200, 201, 202, 204, 206, 211, 212, 213]

# REDCap coding maps
_SEX_MAP = {1.0: "female", 2.0: "male"}
_GENDER_MAP = {1.0: "feman", 2.0: "man", 3.0: "non-binary", 4.0: "other"}
_HAND_MAP = {0.0: "left", 1.0: "right"}
_EYE_MAP = {0.0: "left", 1.0: "right"}
_HEALTH_MAP = {
    1.0: "poor", 2.0: "fair", 3.0: "good", 4.0: "very good",
    5.0: "excellent", 6.0: "excellent",
}
_SMOKE_MAP = {0.0: "no", 1.0: "yes"}


# ---------------------------------------------------------------------------
# PSQI scoring helpers (Buysse et al., 1989 -- 7-component version)
# ---------------------------------------------------------------------------

def _psqi_sleep_quality_component(df: pd.DataFrame) -> pd.Series:
    """Component 1: subjective sleep quality (psqi_6, 0-3 scale directly)."""
    return df["psqi_6"].clip(0, 3)


def _psqi_sleep_latency_component(df: pd.DataFrame) -> pd.Series:
    """Component 2: sleep latency (psqi_2 minutes + psqi_5a difficulty falling asleep)."""
    lat_score = pd.cut(
        df["psqi_2"].fillna(0),
        bins=[-1, 15, 30, 60, np.inf],
        labels=[0, 1, 2, 3],
    ).astype(float)
    diff_score = df["psqi_5a"].fillna(0).clip(0, 3)
    combined = lat_score + diff_score
    return pd.cut(combined, bins=[-1, 0, 2, 4, np.inf], labels=[0, 1, 2, 3]).astype(float)


def _psqi_sleep_duration_component(df: pd.DataFrame) -> pd.Series:
    """Component 3: sleep duration (psqi_4 hours)."""
    return pd.cut(
        df["psqi_4"].fillna(np.nan),
        bins=[-1, 5, 6, 7, np.inf],
        labels=[3, 2, 1, 0],
    ).astype(float)


def _psqi_sleep_efficiency_component(df: pd.DataFrame) -> pd.Series:
    """Component 4: habitual sleep efficiency from bedtime (psqi_1), wake (psqi_3), duration (psqi_4)."""
    def _to_minutes(t: str) -> float:
        try:
            h, m = str(t).strip().split(":")
            return int(h) * 60 + int(m)
        except Exception:
            return np.nan

    bed = df["psqi_1"].apply(_to_minutes)
    wake = df["psqi_3"].apply(_to_minutes)
    # If wake time < bed time, assume wake is next day
    in_bed = (wake - bed).where(wake >= bed, wake - bed + 24 * 60)
    efficiency = (df["psqi_4"] / (in_bed / 60) * 100).clip(0, 100)
    return pd.cut(
        efficiency,
        bins=[-1, 65, 75, 85, np.inf],
        labels=[3, 2, 1, 0],
    ).astype(float)


def _psqi_sleep_disturbance_component(df: pd.DataFrame) -> pd.Series:
    """Component 5: sum of psqi_5b through psqi_5j (9 items, 0-3 each)."""
    items = [f"psqi_5{x}" for x in ["b", "c", "d", "e", "f", "g", "h", "i", "j"]]
    present = [c for c in items if c in df.columns]
    total = df[present].fillna(0).sum(axis=1)
    return pd.cut(total, bins=[-1, 0, 9, 18, np.inf], labels=[0, 1, 2, 3]).astype(float)


def _psqi_sleep_medication_component(df: pd.DataFrame) -> pd.Series:
    """Component 6: sleep medication use (psqi_7, 0-3 scale directly)."""
    return df["psqi_7"].fillna(0).clip(0, 3)


def _psqi_daytime_dysfunction_component(df: pd.DataFrame) -> pd.Series:
    """Component 7: daytime dysfunction (psqi_8 + psqi_9)."""
    combined = df["psqi_8"].fillna(0).clip(0, 3) + df["psqi_9"].fillna(0).clip(0, 3)
    return pd.cut(combined, bins=[-1, 0, 2, 4, np.inf], labels=[0, 1, 2, 3]).astype(float)


def compute_psqi_total(df: pd.DataFrame) -> pd.Series:
    """Compute PSQI global score (0-21) from item columns."""
    components = pd.DataFrame({
        "c1": _psqi_sleep_quality_component(df),
        "c2": _psqi_sleep_latency_component(df),
        "c3": _psqi_sleep_duration_component(df),
        "c4": _psqi_sleep_efficiency_component(df),
        "c5": _psqi_sleep_disturbance_component(df),
        "c6": _psqi_sleep_medication_component(df),
        "c7": _psqi_daytime_dysfunction_component(df),
    })
    return components.sum(axis=1)


# ---------------------------------------------------------------------------
# MCTq: compute MSF_sc (sleep-corrected mid-sleep on free days) as chronotype
# ---------------------------------------------------------------------------

def _time_to_decimal_hours(series: pd.Series) -> pd.Series:
    """Convert HH:MM strings to decimal hours."""
    def _parse(t):
        try:
            h, m = str(t).strip().split(":")
            return int(h) + int(m) / 60
        except Exception:
            return np.nan
    return series.apply(_parse)


def compute_msf_sc(df: pd.DataFrame) -> pd.Series:
    """Compute sleep-corrected mid-sleep on free days (MSFsc) in decimal hours."""
    so_free = _time_to_decimal_hours(df["mumctq_fall_sleep_free"])
    wu_free = _time_to_decimal_hours(df["mumctq_wake_up_free"])
    so_work = _time_to_decimal_hours(df["mumctq_fall_sleep_work"])
    wu_work = _time_to_decimal_hours(df["mumctq_wake_up_work"])

    # Mid-sleep on free days; wrap if sleep onset is after midnight
    sd_free = (wu_free - so_free).where(wu_free >= so_free, wu_free - so_free + 24)
    msf = (so_free + sd_free / 2) % 24

    # Sleep duration on work days
    sd_work = (wu_work - so_work).where(wu_work >= so_work, wu_work - so_work + 24)

    # MSFsc = MSF - (SDfree - SDwork) / 2 when SDfree > SDwork (sleep debt correction)
    diff = (sd_free - sd_work) / 2
    msf_sc = msf - diff.where(sd_free > sd_work, 0.0)
    return msf_sc % 24


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session_info(path: Path = SESSION_INFO_PATH) -> pd.DataFrame:
    """Load and transpose session info Excel into a tidy participant-row DataFrame.

    The Excel file has participants as columns and variables as rows (indexed by
    the 'pid' column). This function transposes the relevant rows into one row
    per participant.

    Notes
    -----
    The 'sex' column in the session info file uses M/F with the coding inverted
    relative to the actual biological sex of participants (confirmed by cross-
    checking all 7 non-missing cases against REDCap lab screening). The mapping
    is therefore applied as: session 'M' -> female, session 'F' -> male.
    """
    raw = pd.read_excel(path)
    wanted_pids = ["sex", "startdate_dl"]
    subset = raw[raw["pid"].isin(wanted_pids)].set_index("pid")

    participant_cols = [c for c in subset.columns if c in PARTICIPANTS]
    subset = subset[participant_cols].T
    subset.index.name = "participant_id"
    subset = subset.reset_index()
    subset["participant_id"] = subset["participant_id"].astype(int)

    # Inverted coding: session 'M' = female, 'F' = male
    _session_sex_map = {"M": "female", "F": "male"}
    subset["sex_session"] = subset["sex"].map(_session_sex_map)
    subset["startdate_dl"] = pd.to_datetime(subset["startdate_dl"])

    return subset[["participant_id", "sex_session", "startdate_dl"]]


def load_lab_screening(path: Path = LAB_SCREENING_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["participant_id"].isin(PARTICIPANTS)].copy()

    df["sex"] = df["demo_sex"].map(_SEX_MAP)
    df["height_cm"] = df["antrop_height"]
    df["weight_kg"] = df["antrop_weight"]
    df["bmi"] = (df["weight_kg"] / (df["height_cm"] / 100) ** 2).round(1)
    df["dominant_hand"] = df["prep_dominant_hand"].map(_HAND_MAP)
    df["dominant_eye"] = df["prep_dominant_eye"].map(_EYE_MAP)
    df["_dob"] = pd.to_datetime(df["demo_dob"], format="mixed")

    return df[
        ["participant_id", "sex", "_dob", "height_cm", "weight_kg",
         "bmi", "dominant_hand", "dominant_eye"]
    ]


def load_online_screening(path: Path = ONLINE_SCREENING_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["participant_id"] = df["participant_id"].astype("Int64")
    df = df[df["participant_id"].isin(PARTICIPANTS)].copy()

    df["gender"] = df["screen_gender"].map(_GENDER_MAP)
    df["health_self_rated"] = df["screen_health"].map(_HEALTH_MAP)
    df["smoker"] = df["screen_smoking"].map(_SMOKE_MAP)

    df["ess_score"] = df["ess_score"]
    df["audit_score"] = df["audit_score"]
    df["srq_revised_score"] = df["srq_revised_score"]
    df["psqi_total"] = compute_psqi_total(df)
    df["msf_sc_h"] = compute_msf_sc(df).round(2)

    return df[
        ["participant_id", "gender", "health_self_rated", "smoker",
         "ess_score", "audit_score", "srq_revised_score", "psqi_total", "msf_sc_h"]
    ]


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def _build_demographics_latex(
    df: pd.DataFrame,
    label: str = "tab:participant_demographics",
    include_audit: bool = True,
) -> str:
    """Build a booktabs participant demographics table."""
    rows = []
    for _, row in df.sort_values("participant_id").iterrows():
        def _fmt(v, decimals=1):
            if isinstance(v, float) and np.isnan(v):
                return "--"
            if isinstance(v, (int, np.integer)):
                return str(v)
            if isinstance(v, float):
                return f"{v:.{decimals}f}"
            return str(v)

        cells = [
            str(int(row["participant_id"])),
            _fmt(row["age_years"]),
            str(row["sex"]),
            str(row["gender"]),
            _fmt(row["height_cm"]),
            _fmt(row["weight_kg"]),
            _fmt(row["bmi"]),
            str(row.get("dominant_hand", "--")),
            str(row.get("dominant_eye", "--")),
            str(row.get("smoker", "--")),
            str(row.get("health_self_rated", "--")),
            _fmt(row.get("ess_score", np.nan), 0),
        ]
        if include_audit:
            cells.append(_fmt(row.get("audit_score", np.nan), 0))
        cells += [
            _fmt(row.get("srq_revised_score", np.nan), 0),
            _fmt(row.get("psqi_total", np.nan), 0),
            _fmt(row.get("msf_sc_h", np.nan), 2),
        ]
        rows.append(" & ".join(cells) + r" \\")

    audit_caption = (
        r" AUDIT = Alcohol Use Disorders Identification Test (0--40, >=8 hazardous);"
        if include_audit else ""
    )
    audit_col_header = r" & \textit{AUDIT}" if include_audit else ""
    n_cols = 16 if include_audit else 15
    col_spec = "r" * n_cols

    lines = [
        r"\begin{table*}",
        r"\centering",
        rf"\caption{{\label{{{label}}}Participant demographics and questionnaire scores."
        r" ESS = Epworth Sleepiness Scale (0--24, >10 indicates excessive daytime sleepiness);"
        + audit_caption +
        r" SRQ = Sleep Regularity Questionnaire (revised score, higher = more regular);"
        r" PSQI = Pittsburgh Sleep Quality Index (0--21, >5 poor sleep quality);"
        r" MSF$_{\text{sc}}$ = sleep-corrected mid-sleep on free days (MCTq chronotype, hours).}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        r"\textit{ID} & \textit{Age} & \textit{Sex} & \textit{Gender}"
        r" & \textit{Height (cm)} & \textit{Weight (kg)} & \textit{BMI}"
        r" & \textit{Hand} & \textit{Eye}"
        r" & \textit{Smoker} & \textit{Health}"
        r" & \textit{ESS}" + audit_col_header + r" & \textit{SRQ} & \textit{PSQI}"
        r" & \textit{MSF$_{\text{sc}}$ [h]} \\",
        r"\bottomrule",# r"\midrule",
    ] + rows + [
        #r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary table (mean +/- SD by sex and overall)
# ---------------------------------------------------------------------------

_NUMERIC_COLS = [
    ("age_years",         "Age [years]"),
    ("height_cm",         "Height [cm]"),
    ("weight_kg",         "Weight [kg]"),
    ("bmi",               "BMI"),
    ("ess_score",         "ESS"),
    ("audit_score",       "AUDIT"),
    ("srq_revised_score", "SRQ (revised)"),
    ("psqi_total",        "PSQI"),
    ("msf_sc_h",          r"MSF$_{\text{sc}}$ [h]"),
]


def _build_summary_latex(
    df: pd.DataFrame,
    label: str = "tab:tab_participant_demographics_summary",
    include_range: bool = True,
    include_audit: bool = True,
) -> str:
    """Build a mean +/- SD summary table stratified by sex and overall."""

    def _ms(series: pd.Series, decimals: int = 1) -> str:
        series = series.dropna()
        if series.empty:
            return "--"
        return rf"{series.mean():.{decimals}f} $\pm$ {series.std():.{decimals}f}"

    def _range(series: pd.Series, decimals: int = 1) -> str:
        series = series.dropna()
        if series.empty:
            return "--"
        return rf"[{series.min():.{decimals}f}; {series.max():.{decimals}f}]"

    groups = {
        "male":   df[df["sex"] == "male"],
        "female": df[df["sex"] == "female"],
        "total":  df,
    }
    n_counts = {g: len(sub) for g, sub in groups.items()}

    header_groups = (
        rf"\textit{{Male}} ($n={n_counts['male']}$)"
        rf" & \textit{{Female}} ($n={n_counts['female']}$)"
        rf" & \textit{{Total}} ($n={n_counts['total']}$)"
        + (r" & \textit{Range (total)}" if include_range else "")
    )

    active_cols = [
        (col, label_str) for col, label_str in _NUMERIC_COLS
        if col != "audit_score" or include_audit
    ]

    rows = []
    for col, label_str in active_cols:
        decimals = 2 if col == "msf_sc_h" else 1
        label_cell = label_str + r"$^{*}$" if (col == "audit_score" and include_audit) else label_str
        cells = (
            [label_cell]
            + [_ms(groups[g][col], decimals) for g in ("male", "female", "total")]
            + ([_range(df[col], decimals)] if include_range else [])
        )
        rows.append(" & ".join(cells) + r" \\")

    n_data_cols = (5 if include_range else 4)
    audit_caption = (
        r" AUDIT = Alcohol Use Disorders Identification Test;"
        if include_audit else ""
    )

    footer_lines = []
    if include_audit:
        audit_missing = df.loc[df["audit_score"].isna()]
        n_miss = len(audit_missing)
        n_miss_male = int((audit_missing["sex"] == "male").sum())
        n_miss_female = int((audit_missing["sex"] == "female").sum())
        audit_missing_str = (
            f"{n_miss} participant(s) ({n_miss_female} female, {n_miss_male} male)"
        )
        footer_lines.append(
            rf"\multicolumn{{{n_data_cols}}}{{l}}{{\footnotesize $^{{*}}$AUDIT data missing for {audit_missing_str}.}} \\"
        )

    lines = [
        r"\begin{table}",
        r"\centering",
        rf"\caption{{\label{{{label}}}Demographic and questionnaire summary by sex."
        r" Values are mean $\pm$ SD; the final column gives the total-sample range as [min; max]."
        r" Age is given in decimal years."
        r" ESS = Epworth Sleepiness Scale;" + audit_caption +
        r" SRQ = Sleep Regularity Questionnaire (revised);"
        r" PSQI = Pittsburgh Sleep Quality Index;"
        r" MSF$_{\text{sc}}$ = sleep-corrected mid-sleep on free days (chronotype).}",
        r"\begin{tabular}{lrrrr}" if include_range else r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textit{Variable} & " + header_groups + r" \\",
        r"\midrule",
    ] + rows + [
        r"\midrule",
    ] + footer_lines + [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _build_summary_csv(df: pd.DataFrame, include_audit: bool = True) -> pd.DataFrame:
    """Return a tidy mean/SD summary DataFrame with columns: variable, male, female, total."""
    groups = {
        "male":   df[df["sex"] == "male"],
        "female": df[df["sex"] == "female"],
        "total":  df,
    }
    active_cols = [
        (col, label_str) for col, label_str in _NUMERIC_COLS
        if col != "audit_score" or include_audit
    ]
    records = []
    for col, label_str in active_cols:
        row = {"variable": label_str}
        for g, sub in groups.items():
            s = sub[col].dropna()
            row[f"{g}_mean"] = round(s.mean(), 3) if not s.empty else float("nan")
            row[f"{g}_sd"] = round(s.std(), 3) if not s.empty else float("nan")
            row[f"{g}_n"] = int(s.notna().sum())
        records.append(row)
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_dir: Path = DEFAULT_RESULTS_DIR, include_audit: bool = True) -> pd.DataFrame:
    lab = load_lab_screening()
    online = load_online_screening()
    session = load_session_info()

    df = lab.merge(online, on="participant_id", how="left")
    df = df.merge(session, on="participant_id", how="left")

    # Fill missing sex from session info (inverted-coding already resolved in load_session_info)
    df["sex"] = df["sex"].fillna(df["sex_session"])
    df = df.drop(columns=["sex_session"])

    # Age at study entry: years elapsed from DOB to startdate_dl
    df["age_years"] = (
        (df["startdate_dl"] - df["_dob"]).dt.days / 365.25
    ).round(1)
    df = df.drop(columns=["_dob", "startdate_dl"])

    df = df.sort_values("participant_id").reset_index(drop=True)

    out_dir = output_dir / "tables" / "participant_info"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "participant_demographics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # LaTeX table
    tex = _build_demographics_latex(df, include_audit=include_audit)
    tex_path = out_dir / "tab_participant_demographics.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"Saved LaTeX table: {tex_path}")

    # Summary table (mean +/- SD by sex and overall)
    summary_df = _build_summary_csv(df, include_audit=include_audit)
    summary_csv_path = out_dir / "participant_demographics_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved summary CSV: {summary_csv_path}")

    summary_tex = _build_summary_latex(df, include_range=True, include_audit=include_audit)
    summary_tex_path = out_dir / "tab_participant_demographics_summary.tex"
    summary_tex_path.write_text(summary_tex, encoding="utf-8")
    print(f"Saved summary LaTeX table: {summary_tex_path}")

    # LaTeX \def variables for summary (row key = variable name -> use group columns as vars)
    vars_summary_df = summary_df.copy()
    # Produce clean ASCII keys suitable for \def macro names
    _label_to_key = {
        label_str: col.replace("_", " ").title().replace(" ", "")
        for col, label_str in _NUMERIC_COLS
    }
    vars_summary_df["variable"] = vars_summary_df["variable"].map(_label_to_key)
    vars_summary_tex = format_latex_variables(
        vars_summary_df, row_key_col="variable", prefix="DemoSummary"
    )
    vars_summary_path = out_dir / "vars_participant_demographics_summary.tex"
    vars_summary_path.write_text(vars_summary_tex, encoding="utf-8")
    print(f"Saved summary LaTeX variables: {vars_summary_path}")

    # LaTeX \def variables -- one row per participant, row key = participant_id
    vars_df = df.copy()
    vars_df["participant_id"] = vars_df["participant_id"].astype(str)
    # Drop string columns that don't translate cleanly to \def values
    vars_df = vars_df.drop(columns=["sex", "gender", "dominant_hand", "dominant_eye",
                                     "smoker", "health_self_rated"], errors="ignore")
    tex_vars = format_latex_variables(vars_df, row_key_col="participant_id",
                                      prefix="Demo")
    vars_path = out_dir / "vars_participant_demographics.tex"
    vars_path.write_text(tex_vars, encoding="utf-8")
    print(f"Saved LaTeX variables: {vars_path}")

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-audit",
        dest="include_audit",
        action="store_false",
        default=True,
        help="Exclude AUDIT score from tables and footnotes.",
    )
    args = parser.parse_args()
    result = main(include_audit=args.include_audit)
    print(result.to_string(index=False))
