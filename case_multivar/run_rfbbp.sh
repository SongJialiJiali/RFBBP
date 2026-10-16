#!/usr/bin/env bash
set -euo pipefail

# Unified runner for the locked primary RFBBP analysis and its prior
# sensitivity analyses.
#
# Usage:
#   bash case_multivar/run_rfbbp.sh primary
#   bash case_multivar/run_rfbbp.sh S1        # likewise S2 ... S5
#   bash case_multivar/run_rfbbp.sh missing   # S1, S3, and S4
#   bash case_multivar/run_rfbbp.sh sensitivity
#   bash case_multivar/run_rfbbp.sh all       # primary plus S1 ... S5
#
# With no argument, all analyses are run. Set SKIP_NETCDF=1 to retain only
# summary, diagnostic, metadata, and aggregate result files.

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${SCRIPT_DIR}/analysis_final/analysis_datasets"
PRIMARY_OUTPUT_DIR="${SCRIPT_DIR}/analysis_final/RFBBP_result"
PRIMARY_TABLE_DIR="${SCRIPT_DIR}/analysis_final/tables"
SENSITIVITY_OUTPUT_DIR="${SCRIPT_DIR}/analysis_final/RFBBP_result/prior_sensitivity"
SENSITIVITY_TABLE_DIR="${SCRIPT_DIR}/analysis_final/tables/prior_sensitivity"

NETCDF_ARGS=()
if [[ "${SKIP_NETCDF:-0}" == "1" ]]; then
  NETCDF_ARGS=(--skip-netcdf)
fi

COMMON_MODEL_ARGS=(
  --adjustment sofa_adjusted
  --standardize-w none
  --survival-model pem
  --draws 2000
  --chains 4
  --cores 4
  --include-databases mimic_iv eicu
  --no-center-filter
  --min-center-n 50
  --min-center-events 10
  --min-center-nonevents 10
  --betax-sigma0 2.5
  --betaz-sigma0 1.0
  --beta0-first-mu0 0.0
  --beta0-first-sigma0 1.0
  --mu-a-sigma0 0.5
  --tau-a-sigma0 0.5
  --mu-l-sigma0 0.5
  --tau-l-sigma0 0.5
  --sigma-w-prior invgamma
  --sigma-w2-alpha0 3
  --sigma-w2-gamma0 1
  --sigma-delta-alpha0 3
  --sigma-delta-gamma0 0.5
)

run_primary() {
  echo "============================================================"
  echo "Primary full-data linear RFBBP analyses"
  echo "============================================================"

  python run_rfbbp.py \
    --variables Hemoglobin_hour_24 Lactate_hour_24 \
    "${COMMON_MODEL_ARGS[@]}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${PRIMARY_OUTPUT_DIR}" \
    --table-dir "${PRIMARY_TABLE_DIR}" \
    --tune 2000 \
    --target-accept 0.95 \
    "${NETCDF_ARGS[@]}"

  # Creatinine retains the more conservative MCMC controls used in the
  # locked primary analysis.
  python run_rfbbp.py \
    --variables Creatinine_hour_24 \
    "${COMMON_MODEL_ARGS[@]}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${PRIMARY_OUTPUT_DIR}" \
    --table-dir "${PRIMARY_TABLE_DIR}" \
    --tune 2000 \
    --target-accept 0.95 \
    --seed 3002 \
    "${NETCDF_ARGS[@]}"
}

run_sensitivity_model() {
  local variable="$1"
  local label_prefix="$2"
  local scenario_label="$3"
  local tune="$4"
  local target_accept="$5"
  shift 5

  python run_rfbbp.py \
    --variables "${variable}" \
    "${COMMON_MODEL_ARGS[@]}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${SENSITIVITY_OUTPUT_DIR}" \
    --table-dir "${SENSITIVITY_TABLE_DIR}" \
    --tune "${tune}" \
    --target-accept "${target_accept}" \
    --run-label "${label_prefix}_${scenario_label}" \
    "${NETCDF_ARGS[@]}" \
    "$@"
}

run_sensitivity_scenario() {
  local scenario="$1"
  local scenario_label
  local lactate_scenario_label
  local scenario_args=()

  case "${scenario}" in
    S1)
      scenario_label="prior_S1_more_diffuse_calibration"
      scenario_args=(--mu-a-sigma0 1.0 --mu-l-sigma0 1.0)
      ;;
    S2)
      scenario_label="prior_NS2_stronger_hier_shrinkage"
      scenario_args=(--tau-a-sigma0 0.25 --tau-l-sigma0 0.25)
      ;;
    S3)
      scenario_label="prior_S3_weaker_hier_shrinkage"
      scenario_args=(--tau-a-sigma0 1.0 --tau-l-sigma0 1.0)
      ;;
    S4)
      scenario_label="prior_S4_hn_sigmaw1"
      scenario_args=(--sigma-w-prior halfnormal --sigma-w-scale0 1.0)
      ;;
    S5)
      scenario_label="prior_NS4_betax_sigma1"
      scenario_args=(--betax-sigma0 1.0)
      ;;
    *)
      echo "Unknown prior-sensitivity scenario: ${scenario}" >&2
      return 2
      ;;
  esac

  echo "============================================================"
  echo "Prior sensitivity analysis ${scenario}"
  echo "============================================================"

  lactate_scenario_label="${scenario_label}"
  if [[ "${scenario}" == "S1" ]]; then
    # The final notebook intentionally selects the converged S1 lactate rerun.
    lactate_scenario_label="${scenario_label}_rerun"
  fi
  run_sensitivity_model Hemoglobin_hour_24 hb "${scenario_label}" 2000 0.95 "${scenario_args[@]}"
  run_sensitivity_model Lactate_hour_24 lactate "${lactate_scenario_label}" 2000 0.95 "${scenario_args[@]}"
  run_sensitivity_model Creatinine_hour_24 creatinine "${scenario_label}" 2000 0.95 "${scenario_args[@]}"
}

requested="${1:-all}"
requested="${requested^^}"

case "${requested}" in
  PRIMARY)
    run_primary
    ;;
  S1|S2|S3|S4|S5)
    run_sensitivity_scenario "${requested}"
    ;;
  MISSING)
    for scenario in S1 S3 S4; do
      run_sensitivity_scenario "${scenario}"
    done
    ;;
  SENSITIVITY)
    for scenario in S1 S2 S3 S4 S5; do
      run_sensitivity_scenario "${scenario}"
    done
    ;;
  ALL)
    run_primary
    for scenario in S1 S2 S3 S4 S5; do
      run_sensitivity_scenario "${scenario}"
    done
    ;;
  *)
    echo "Usage: $0 [primary|S1|S2|S3|S4|S5|missing|sensitivity|all]" >&2
    exit 2
    ;;
esac

echo "============================================================"
echo "Finished RFBBP run selection: ${requested}"
echo "============================================================"
