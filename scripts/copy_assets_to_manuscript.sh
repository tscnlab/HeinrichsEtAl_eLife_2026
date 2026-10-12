#!/usr/bin/env bash
# Copy generated assets from results/ into the manuscript folder.
# Run from the repository root.
#
# Usage: ./copy_assets_to_manuscript.sh [--manuscript <dir>] [--fig] [--tab] [--tex] [--all] [--help|-h]
#
# Options:
#   --manuscript <dir>  target manuscript directory (default: manuscript/20260629_civibe_pupillometry)
#
# Flags (one or more, in any order):
#   --fig        copy figures  (results/figures/  -> <manuscript>/figures/)
#   --tab        copy tables   (results/tables/   -> <manuscript>/tables/)
#   --tex        copy tex vars (results/tex/      -> <manuscript>/tex/)
#   --all        copy figures, tables, and tex vars
#   --help | -h  show help
#
# Each run writes <manuscript>/asset_manifest.txt (date, git commit, sha256 of
# every copied file) so a built PDF can be traced to its inputs.

set -euo pipefail

DEFAULT_MANUSCRIPT="manuscript_elife"

usage() {
    cat <<EOF
Usage: $(basename "$0") [--dst <dir>] [--fig] [--tab] [--tex] [--all] [--help|-h]

Copy generated assets from results/ into the manuscript folder.
Run from the repository root.

Options:
  --dst <dir>  Target manuscript directory (default: $DEFAULT_MANUSCRIPT)

Flags (one or more, in any order):
  --fig        Copy figures   results/figures/ -> <manuscript>/figures/
  --tab        Copy tables    results/tables/  -> <manuscript>/tables/
  --tex        Copy tex vars  results/tex/     -> <manuscript>/tex/
  --all        Copy figures, tables, and tex vars
  --help | -h  Show this help message

Examples:
  $(basename "$0") --fig --tex                          # copy figures and tex variable snippets
  $(basename "$0") --tab                                # copy tables only
  $(basename "$0") --all                                # copy everything
  $(basename "$0") --dst my/manuscript --all     # copy everything to a custom directory
EOF
}

if [[ $# -eq 0 ]]; then
    usage
    exit 0
fi

ARGS="$*"
RESULTS="results"
# Destination paths (relative to $DST) of every copied file, for the manifest
COPIED=()
DST="$DEFAULT_MANUSCRIPT"

DO_FIG=0
DO_TAB=0
DO_TEX=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dst)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "--dst requires a directory argument"
                exit 1
            fi
            DST="$2"
            shift 2
            ;;
        --fig)  DO_FIG=1; shift ;;
        --tab)  DO_TAB=1; shift ;;
        --tex)  DO_TEX=1; shift ;;
        --all)  DO_FIG=1; DO_TAB=1; DO_TEX=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *)
            echo "Unknown flag: $1"
            echo ""
            usage
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

copy_fig() {
    local src="$1"
    local dest="$DST/figures/$2"
    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
    COPIED+=("figures/$2")
    echo "  $src -> $dest"
}

if [[ $DO_FIG -eq 1 ]]; then
    echo "=== Copying figures ==="

    # aggtraces
    copy_fig "$RESULTS/figures/aggtraces_by_condition/fig_global_condition_means.png" \
        "aggtraces_by_condition/fig_global_condition_means.png"

    # pupil cosinor
    copy_fig "$RESULTS/figures/pupil_cosinor/participant_201/fig_cosinor_2step_fits_with_overlay_201.png" \
        "pupil_cosinor/fig_cosinor_2step_fits_with_overlay_201.png"
    copy_fig "$RESULTS/figures/pupil_cosinor/fig_pupil_cosinor_decomposition_202.png" \
        "pupil_cosinor/fig_pupil_cosinor_decomposition_202.png"
    copy_fig "$RESULTS/figures/pupil_cosinor/fig_pupil_cosinor_decomposition_average.png" \
        "pupil_cosinor/fig_pupil_cosinor_decomposition_average.png"
    copy_fig "$RESULTS/figures/pupil_cosinor/fig_pupil_cosinor_parameters.png" \
        "pupil_cosinor/fig_pupil_cosinor_parameters.png"
    # Exemplary single participant (202); main.tex references the _202 name.
    copy_fig "$RESULTS/figures/pupil_cosinor/fig_pupil_cosinor_parameters_202.png" \
        "pupil_cosinor/fig_pupil_cosinor_parameters_202.png"

    # pupil cosinor diagnostics
    copy_fig "$RESULTS/figures/pupil_cosinor_diagnostics/fig_pupil_cosinor_diagnostics_combined.png" \
        "pupil_cosinor_diagnostics/fig_pupil_cosinor_diagnostics_combined.png"

    # saliva cosinor
    copy_fig "$RESULTS/figures/saliva_cosinor/fig_saliva_cosinor_summary.png" \
        "saliva_cosinor/fig_saliva_cosinor_summary.png"
    copy_fig "$RESULTS/figures/saliva_cosinor/fig_metric_vs_tau.png" \
        "saliva_cosinor/fig_metric_vs_tau.png"
    copy_fig "$RESULTS/figures/saliva_cosinor/fig_participants_best_fit.png" \
        "saliva_cosinor/fig_participants_best_fit.png"

    # circadian cosinor (amplitude -- fitted vs empirical)
    # The combined all-conditions panel is the figure used in the manuscript;
    # only flux and mel are additionally shown as exemplary per-condition fits.
    copy_fig "$RESULTS/figures/circadian_cosinor/amplitude/fig_fitted_vs_empirical_all_conditions.png" \
        "circadian_cosinor/amplitude/fig_fitted_vs_empirical_all_conditions.png"
    for cond in flux mel; do
        copy_fig "$RESULTS/figures/circadian_cosinor/amplitude/fig_fitted_vs_empirical_${cond}.png" \
            "circadian_cosinor/amplitude/fig_fitted_vs_empirical_${cond}.png"
    done

    # circadian cosinor (tonic_min -- fitted vs empirical)
    # Combined panel plus l-m and mel as exemplary per-condition fits.
    copy_fig "$RESULTS/figures/circadian_cosinor/tonic_min/fig_fitted_vs_empirical_all_conditions.png" \
        "circadian_cosinor/tonic_min/fig_fitted_vs_empirical_all_conditions.png"
    for cond in l-m mel; do
        copy_fig "$RESULTS/figures/circadian_cosinor/tonic_min/fig_fitted_vs_empirical_${cond}.png" \
            "circadian_cosinor/tonic_min/fig_fitted_vs_empirical_${cond}.png"
    done

    # circadian cosinor diagnostics (amplitude)
    for cond in flux l-m lms mel s; do
        copy_fig "$RESULTS/figures/circadian_cosinor_diagnostics/amplitude/fig_amplitude_diagnostics_${cond}.png" \
            "circadian_cosinor_diagnostics/amplitude/fig_amplitude_diagnostics_${cond}.png"
    done

    # circadian cosinor diagnostics (tonic_min)
    for cond in flux l-m lms mel s; do
        copy_fig "$RESULTS/figures/circadian_cosinor_diagnostics/tonic_min/fig_tonic_min_diagnostics_${cond}.png" \
            "circadian_cosinor_diagnostics/tonic_min/fig_tonic_min_diagnostics_${cond}.png"
    done

    # protocol-specified outcome plr_ratio (si_pipeline.py)
    # Step 3b endpoint figures, then step 5c figures beside the primary outcomes.
    for fig in endpoint ordering ratios; do
        copy_fig "$RESULTS/figures/plr_ratio/fig_plr_ratio_${fig}.png" \
            "plr_ratio/fig_plr_ratio_${fig}.png"
    done
    copy_fig "$RESULTS/figures/circadian_cosinor/plr_ratio/fig_plr_ratio_bf_comparison.png" \
        "circadian_cosinor/plr_ratio/fig_plr_ratio_bf_comparison.png"
    copy_fig "$RESULTS/figures/circadian_cosinor/plr_ratio/fig_fitted_vs_empirical_mel.png" \
        "circadian_cosinor/plr_ratio/fig_fitted_vs_empirical_mel.png"

    echo ""
fi

# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------

copy_tab() {
    local src="$1"
    local dest="$DST/tables/$2"
    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
    COPIED+=("tables/$2")
    echo "  $src -> $dest"
}

if [[ $DO_TAB -eq 1 ]]; then
    echo "=== Copying tables ==="

    # participant info
    copy_tab "$RESULTS/tables/participant_info/tab_participant_demographics_summary.tex" \
        "participant_info/tab_participant_demographics_summary.tex"

    # saliva cosinor
    copy_tab "$RESULTS/tables/saliva_cosinor/saliva_cosinor_best_fit.tex" \
        "saliva_cosinor/saliva_cosinor_best_fit.tex"

    # descriptives
    copy_tab "$RESULTS/tables/descriptives/pupil_metrics.tex" \
        "descriptives/pupil_metrics.tex"
    copy_tab "$RESULTS/descriptives/tab_pupil_descriptive_metrics_selected.tex" \
        "descriptives/tab_pupil_descriptive_metrics_selected.tex"

    # pupil cosinor
    copy_tab "$RESULTS/tables/pupil_cosinor/tab_pupil_cosinor_2step_tonic.tex" \
        "pupil_cosinor/tab_pupil_cosinor_2step_tonic.tex"
    copy_tab "$RESULTS/tables/pupil_cosinor/tab_pupil_cosinor_2step_phasic.tex" \
        "pupil_cosinor/tab_pupil_cosinor_2step_phasic.tex"

    # circadian cosinor (amplitude)
    # tab_condition_sample_sizes is not copied: N_obs/N_clust are identical across
    # outcome metrics and already reported in the per-outcome model summary tables.
    copy_tab "$RESULTS/tables/circadian_cosinor/amplitude/tab_parameter_estimates_amplitude.tex" \
        "circadian_cosinor/amplitude/tab_parameter_estimates_amplitude.tex"
    copy_tab "$RESULTS/tables/circadian_cosinor/amplitude/tab_model_summary_amplitude.tex" \
        "circadian_cosinor/amplitude/tab_model_summary_amplitude.tex"
    copy_tab "$RESULTS/tables/circadian_cosinor/tab_model_summary.tex" \
        "circadian_cosinor/tab_model_summary.tex"

    # circadian cosinor (tonic_min -- secondary outcome)
    copy_tab "$RESULTS/tables/circadian_cosinor/tonic_min/tab_parameter_estimates_tonic_min.tex" \
        "circadian_cosinor/tonic_min/tab_parameter_estimates_tonic_min.tex"
    copy_tab "$RESULTS/tables/circadian_cosinor/tonic_min/tab_model_summary_tonic_min.tex" \
        "circadian_cosinor/tonic_min/tab_model_summary_tonic_min.tex"

    # circadian cosinor (exploratory outcomes -- supplement only)
    # All remaining outcome parameters except amplitude and tonic_min; each has a
    # parameter-estimates and a model-summary table, inputted by supplement.tex.
    for outcome in baseline phase amplitude_2f phase_2f log_amplitude log_amplitude_2f; do
        copy_tab "$RESULTS/tables/circadian_cosinor/${outcome}/tab_parameter_estimates_${outcome}.tex" \
            "circadian_cosinor/${outcome}/tab_parameter_estimates_${outcome}.tex"
        copy_tab "$RESULTS/tables/circadian_cosinor/${outcome}/tab_model_summary_${outcome}.tex" \
            "circadian_cosinor/${outcome}/tab_model_summary_${outcome}.tex"
    done

    # circadian cosinor diagnostics
    # One combined table per outcome: diagnostics as rows, conditions as columns,
    # replacing the former per-group (bf/ppc/residuals/acf/convergence) tables.
    for outcome in amplitude tonic_min baseline phase amplitude_2f phase_2f log_amplitude log_amplitude_2f; do
        copy_tab "$RESULTS/tables/circadian_cosinor_diagnostics/${outcome}/tab_diagnostics_summary_${outcome}.tex" \
            "circadian_cosinor_diagnostics/${outcome}/tab_diagnostics_summary_${outcome}.tex"
    done

    # protocol-specified outcome plr_ratio (appendix)
    copy_tab "$RESULTS/tables/circadian_cosinor/plr_ratio/tab_parameter_estimates_plr_ratio.tex" \
        "circadian_cosinor/plr_ratio/tab_parameter_estimates_plr_ratio.tex"
    copy_tab "$RESULTS/tables/circadian_cosinor_diagnostics/plr_ratio/tab_diagnostics_summary_plr_ratio.tex" \
        "circadian_cosinor_diagnostics/plr_ratio/tab_diagnostics_summary_plr_ratio.tex"

    echo ""
fi

# ---------------------------------------------------------------------------
# tex variable snippets
# ---------------------------------------------------------------------------

copy_tex() {
    local src="$1"
    local dest="$DST/tex/$2"
    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
    COPIED+=("tex/$2")
    echo "  $src -> $dest"
}

if [[ $DO_TEX -eq 1 ]]; then
    echo "=== Copying tex snippets ==="

    copy_tex "$RESULTS/tex/circadian_cosinor/amplitude/vars_circadian_cosinor_parameters.tex" \
        "circadian_cosinor/amplitude/vars_circadian_cosinor_parameters.tex"
    copy_tex "$RESULTS/tex/circadian_cosinor/amplitude/vars_circadian_model_summary.tex" \
        "circadian_cosinor/amplitude/vars_circadian_model_summary.tex"
    copy_tex "$RESULTS/tex/circadian_cosinor/tonic_min/vars_circadian_cosinor_parameters.tex" \
        "circadian_cosinor/tonic_min/vars_circadian_cosinor_parameters.tex"
    copy_tex "$RESULTS/tex/circadian_cosinor/tonic_min/vars_circadian_model_summary.tex" \
        "circadian_cosinor/tonic_min/vars_circadian_model_summary.tex"
    copy_tex "$RESULTS/tex/circadian_cosinor_diagnostics/amplitude/vars_diagnose_circadian_amplitude.tex" \
        "circadian_cosinor_diagnostics/amplitude/vars_diagnose_circadian_amplitude.tex"
    copy_tex "$RESULTS/tex/circadian_cosinor_diagnostics/tonic_min/vars_diagnose_circadian_tonic_min.tex" \
        "circadian_cosinor_diagnostics/tonic_min/vars_diagnose_circadian_tonic_min.tex"

    # exploratory outcomes -- macros cited by the supplement
    for outcome in baseline phase amplitude_2f phase_2f log_amplitude log_amplitude_2f; do
        copy_tex "$RESULTS/tex/circadian_cosinor/${outcome}/vars_circadian_cosinor_parameters.tex" \
            "circadian_cosinor/${outcome}/vars_circadian_cosinor_parameters.tex"
        copy_tex "$RESULTS/tex/circadian_cosinor/${outcome}/vars_circadian_model_summary.tex" \
            "circadian_cosinor/${outcome}/vars_circadian_model_summary.tex"
        copy_tex "$RESULTS/tex/circadian_cosinor_diagnostics/${outcome}/vars_diagnose_circadian_${outcome}.tex" \
            "circadian_cosinor_diagnostics/${outcome}/vars_diagnose_circadian_${outcome}.tex"
    done
    # protocol-specified outcome plr_ratio (appendix)
    copy_tex "$RESULTS/tex/plr_ratio/vars_plr_ratio.tex" "plr_ratio/vars_plr_ratio.tex"
    copy_tex "$RESULTS/tex/plr_ratio/vars_plr_ratio_traces.tex" "plr_ratio/vars_plr_ratio_traces.tex"
    copy_tex "$RESULTS/tex/circadian_cosinor/plr_ratio/vars_circadian_cosinor_parameters.tex" \
        "circadian_cosinor/plr_ratio/vars_circadian_cosinor_parameters.tex"
    copy_tex "$RESULTS/tex/circadian_cosinor/plr_ratio/vars_circadian_model_summary.tex" \
        "circadian_cosinor/plr_ratio/vars_circadian_model_summary.tex"
    copy_tex "$RESULTS/tex/circadian_cosinor_diagnostics/plr_ratio/vars_diagnose_circadian_plr_ratio.tex" \
        "circadian_cosinor_diagnostics/plr_ratio/vars_diagnose_circadian_plr_ratio.tex"
    # log-transformed ratio sensitivity check (Appendix 4 table)
    copy_tex "$RESULTS/tex/circadian_cosinor/log_plr_ratio/vars_circadian_cosinor_parameters.tex" \
        "circadian_cosinor/log_plr_ratio/vars_circadian_cosinor_parameters.tex"
    copy_tex "$RESULTS/tex/circadian_cosinor_diagnostics/log_plr_ratio/vars_diagnose_circadian_log_plr_ratio.tex" \
        "circadian_cosinor_diagnostics/log_plr_ratio/vars_diagnose_circadian_log_plr_ratio.tex"

    copy_tex "$RESULTS/tex/pupil_cosinor/vars_pupil_cosinor_2step.tex" \
        "pupil_cosinor/vars_pupil_cosinor_2step.tex"
    copy_tex "$RESULTS/tex/aggregated/vars_trial_counts.tex" \
        "aggregated/vars_trial_counts.tex"
    copy_tex "$RESULTS/tex/saliva_cosinor/vars_saliva_cosinor_summary.tex" \
        "saliva_cosinor/vars_saliva_cosinor_summary.tex"

    echo ""
fi

# ---------------------------------------------------------------------------
# manifest: ties a manuscript build to the code commit and exact assets
# ---------------------------------------------------------------------------

if [[ ${#COPIED[@]} -gt 0 ]]; then
    MANIFEST="$DST/asset_manifest.txt"
    COMMIT="$(git rev-parse --short HEAD)"
    if [[ -n "$(git status --porcelain -- src scripts pipeline.py si_pipeline.py)" ]]; then
        COMMIT="$COMMIT (uncommitted changes in src/scripts/pipelines)"
    fi
    {
        echo "# asset manifest -- written by $(basename "$0")"
        echo "date:   $(date '+%Y-%m-%d %H:%M:%S')"
        echo "commit: $COMMIT"
        echo "flags:  $ARGS"
        echo ""
        (cd "$DST" && shasum -a 256 "${COPIED[@]}")
    } > "$MANIFEST"
    echo "=== Wrote $MANIFEST (${#COPIED[@]} files) ==="
fi

echo "Done."
