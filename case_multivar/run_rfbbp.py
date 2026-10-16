from __future__ import annotations

import os

for name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ.setdefault(name, "1")

for path in [
    "/tmp/arviz",
    "/tmp/xdg_cache_rfbbp",
    "/tmp/matplotlib_rfbbp",
    "/tmp/pytensor_rfbbp",
    "/tmp/aesara_rfbbp",
]:
    os.makedirs(path, exist_ok=True)

os.environ["ARVIZ_DATA"] = "/tmp/arviz"
os.environ["XDG_CACHE_HOME"] = "/tmp/xdg_cache_rfbbp"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_rfbbp"
os.environ["PYTENSOR_FLAGS"] = "base_compiledir=/tmp/pytensor_rfbbp,compile__timeout=600,compile__wait=5"
os.environ["AESARA_FLAGS"] = "base_compiledir=/tmp/aesara_rfbbp"

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# This script is located under case_multivar, while the project source code is
# stored in the sibling top-level src directory. Add src to sys.path so
# RFBBPPyMC can be imported directly from the rfbbp package below.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# The formal analysis covers three core laboratory biomarkers by default and
# uses only the sofa_adjusted datasets here.
DEFAULT_VARIABLES = [
    "Hemoglobin_hour_24",
    "Lactate_hour_24",
    "Creatinine_hour_24",
]
ALLOWED_VARIABLES = [
    "Hemoglobin_hour_24",
    "Lactate_hour_24",
    "Creatinine_hour_24",
    "Heart_rate_hour_24",
    "Temperature_hour_24",
    "MAP_hour_24",
    "Platelet_hour_24",
    "RAR_hour_24",
]
DEFAULT_ADJUSTMENT = "sofa_adjusted"

# Use fixed clinical cut points at days 1, 3, 7, 14, and 28.
# RFBBPPyMC also requires the left endpoint 0 in cutpoints, hence the complete
# sequence 0, 1, 3, 7, 14, and 28 below.
CUTPOINTS = [0.0, 1.0, 3.0, 7.0, 14.0, 28.0]

# Do not filter centers by sample size or event counts by default. Retain the
# thresholds so filtering can be enabled explicitly for sensitivity analyses.
APPLY_CENTER_FILTER = False
MIN_CENTER_N = 50
MIN_CENTER_EVENTS = 10
MIN_CENTER_NONEVENTS = 10

# Include MIMIC-IV and eICU in the formal analysis.
INCLUDE_DATABASES = ("mimic_iv", "eicu")

# Use the same basic covariates for all biomarkers.
BASIC_Z_COLS = ["age_z", "gender_male"]

# The sofa_adjusted analyses use biomarker-specific SOFA adjustment columns.
# Creatinine, platelet count, and MAP exclude their corresponding organ
# subscores; all other biomarkers use the total SOFA score.
SOFA_Z_COL = {
    "Lactate_hour_24": "SOFA_score_hour_24_z",
    "Creatinine_hour_24": "SOFA_no_renal_z",
    "Heart_rate_hour_24": "SOFA_score_hour_24_z",
    "Temperature_hour_24": "SOFA_score_hour_24_z",
    "MAP_hour_24": "SOFA_no_cardiovascular_z",
    "Hemoglobin_hour_24": "SOFA_score_hour_24_z",
    "Platelet_hour_24": "SOFA_no_coagulation_z",
    "RAR_hour_24": "SOFA_score_hour_24_z",
}


def set_runtime_env() -> None:
    """Limit numeric-library threads before PyMC starts multiprocessing."""
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ.setdefault(name, "1")

    # ArviZ/Matplotlib/PyTensor sometimes inherit unwritable home-cache settings.
    # Force writable tmp dirs so PyTensor compilation cannot silently stall on /home.
    os.environ["ARVIZ_DATA"] = "/tmp/arviz"
    os.environ["XDG_CACHE_HOME"] = "/tmp/xdg_cache_rfbbp"
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_rfbbp"
    os.environ["PYTENSOR_FLAGS"] = "base_compiledir=/tmp/pytensor_rfbbp,compile__timeout=600,compile__wait=5"
    os.environ["AESARA_FLAGS"] = "base_compiledir=/tmp/aesara_rfbbp"

    for key in ["ARVIZ_DATA", "XDG_CACHE_HOME", "MPLCONFIGDIR"]:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    Path("/tmp/pytensor_rfbbp").mkdir(parents=True, exist_ok=True)
    Path("/tmp/aesara_rfbbp").mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal hierarchical RFBBP MCMC for prepared analysis datasets."
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["all"],
        help="Variables to run. Use 'all' for all formal analysis variables, or list one/more variable names.",
    )
    parser.add_argument(
        "--adjustment",
        default=DEFAULT_ADJUSTMENT,
        choices=["sofa_adjusted"],
        help="Adjustment dataset suffix. Current formal RFBBP uses sofa_adjusted.",
    )
    parser.add_argument("--draws", type=int, default=2000, help="Posterior draws per chain.")
    parser.add_argument("--tune", type=int, default=2000, help="Tuning iterations per chain.")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--cores", type=int, default=4, help="Number of PyMC worker processes.")
    parser.add_argument("--target-accept", type=float, default=0.95, help="NUTS target acceptance probability.")
    parser.add_argument("--seed", type=int, default=3000, help="Base random seed.")
    parser.add_argument(
        "--standardize-w",
        choices=["none", "global"],
        default="none",
        help="How RFBBP standardizes exposure_analysis before fitting. 'none' uses the analysis-scale W as-is; 'global' uses global z-score.",
    )
    parser.add_argument(
        "--survival-model",
        choices=["pem", "cox", "rcs"],
        default="pem",
        help=(
            "Survival layer for RFBBP. 'pem' uses the current piecewise exponential model; "
            "'cox' uses RFBBPCox with stratified Cox partial likelihood; "
            "'rcs' uses RFBBPRCS with a nonlinear latent-X spline Cox survival layer."
        ),
    )
    parser.add_argument("--min-center-n", type=int, default=MIN_CENTER_N, help="Minimum center sample size.")
    parser.add_argument("--min-center-events", type=int, default=MIN_CENTER_EVENTS, help="Minimum center event count.")
    parser.add_argument("--min-center-nonevents", type=int, default=MIN_CENTER_NONEVENTS, help="Minimum center non-event count.")
    parser.add_argument(
        "--center-filter",
        dest="apply_center_filter",
        action="store_true",
        default=APPLY_CENTER_FILTER,
        help="Apply center n/events/nonevents eligibility filtering after complete-case filtering.",
    )
    parser.add_argument(
        "--no-center-filter",
        dest="apply_center_filter",
        action="store_false",
        help="Disable center n/events/nonevents eligibility filtering after complete-case filtering.",
    )
    parser.add_argument(
        "--include-databases",
        nargs="+",
        choices=["mimic_iv", "eicu"],
        default=list(INCLUDE_DATABASES),
        help="Databases to include before complete-case and center-eligibility filtering.",
    )
    parser.add_argument(
        "--exclude-centers",
        nargs="*",
        default=[],
        help=(
            "Optional center_id values to remove after database, complete-case, "
            "and center-eligibility filtering. Useful for leave-one-center diagnostics."
        ),
    )
    parser.add_argument(
        "--skip-netcdf",
        action="store_true",
        help="Do not save the large ArviZ posterior NetCDF file. Summary, diagnostics, and metadata are still saved.",
    )
    parser.add_argument("--betax-sigma0", type=float, default=2.5, help="Prior SD for beta_x.")
    parser.add_argument("--betaz-sigma0", type=float, default=1.0, help="Prior SD for beta_z coefficients.")
    parser.add_argument("--beta0-first-mu0", type=float, default=0.0, help="Prior mean for the first baseline log-hazard interval.")
    parser.add_argument("--beta0-first-sigma0", type=float, default=1.0, help="Prior SD for the first baseline log-hazard interval.")
    parser.add_argument("--sigma-delta-alpha0", type=float, default=3.0, help="Inverse-Gamma alpha prior for random-walk increment variance.")
    parser.add_argument("--sigma-delta-gamma0", type=float, default=0.5, help="Inverse-Gamma beta prior for random-walk increment variance.")
    parser.add_argument("--mu-a-sigma0", type=float, default=0.5, help="Prior SD for hierarchical calibration intercept mean.")
    parser.add_argument("--tau-a-sigma0", type=float, default=0.5, help="HalfNormal scale for calibration intercept heterogeneity.")
    parser.add_argument("--mu-l-sigma0", type=float, default=0.5, help="Prior SD for hierarchical calibration log-slope mean.")
    parser.add_argument("--tau-l-sigma0", type=float, default=0.5, help="HalfNormal scale for calibration log-slope heterogeneity.")
    parser.add_argument(
        "--sigma-w-prior",
        choices=["invgamma", "halfnormal", "halfcauchy"],
        default="invgamma",
        help="Prior family for center-specific measurement-error SD.",
    )
    parser.add_argument("--sigma-w-scale0", type=float, default=1.0, help="Scale for halfnormal/halfcauchy sigma_w priors.")
    parser.add_argument("--sigma-w2-alpha0", type=float, default=3.0, help="Inverse-Gamma alpha prior for sigma_w^2.")
    parser.add_argument("--sigma-w2-gamma0", type=float, default=1.0, help="Inverse-Gamma beta prior for sigma_w^2.")
    parser.add_argument(
        "--rcs-knots",
        nargs="+",
        type=float,
        default=[-1.64, -0.43, 0.43, 1.64],
        help="Knots on the latent X scale for --survival-model rcs.",
    )
    parser.add_argument(
        "--rcs-coef-sigma0",
        type=float,
        default=1.0,
        help="Prior SD for RCS latent-X spline coefficients.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for per-variable RFBBP files. Default is chosen from --standardize-w.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing <variable>_<adjustment>.csv analysis datasets.",
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=None,
        help="Directory for aggregate RFBBP result and diagnostic CSV files.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help=(
            "Optional label appended to per-variable and aggregate output files. "
            "If omitted, a label is generated from model settings."
        ),
    )
    return parser.parse_args()


def w_label(standardize_w: str) -> str:
    """Return the file-name label for the requested W scale."""
    if standardize_w == "none":
        return "raw"
    if standardize_w == "global":
        return "global"
    raise ValueError(f"Unsupported standardize_w={standardize_w!r}.")


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve 'all' variables and the default output directory after parsing CLI options."""
    variables = list(args.variables)
    if len(variables) == 1 and variables[0].lower() == "all":
        args.variables = DEFAULT_VARIABLES.copy()
    elif any(v.lower() == "all" for v in variables):
        raise ValueError("--variables all cannot be combined with explicit variable names.")
    else:
        unknown = [v for v in variables if v not in ALLOWED_VARIABLES]
        if unknown:
            raise ValueError(f"Unknown variable(s): {unknown}. Allowed variables: {ALLOWED_VARIABLES}")
        args.variables = variables

    if args.output_dir is None:
        model_dir_token = {
            "pem": "rfbbp",
            "cox": "rfbbpcox",
            "rcs": "rfbbprcs",
        }[args.survival_model]
        args.output_dir = CASE_DIR / f"analysis_simple/output/direct_{model_dir_token}_hierarchical_all_metrics_w_{w_label(args.standardize_w)}"
    if args.data_dir is None:
        args.data_dir = CASE_DIR / "analysis_simple/output/analysis_datasets"
    if args.table_dir is None:
        args.table_dir = CASE_DIR / "analysis_simple/output/tables"

    positive_int_fields = ["draws", "tune", "chains", "cores", "min_center_n", "min_center_events", "min_center_nonevents"]
    for field in positive_int_fields:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    if args.cores > args.chains:
        args.cores = args.chains

    positive_float_fields = [
        "target_accept",
        "betax_sigma0",
        "betaz_sigma0",
        "beta0_first_sigma0",
        "sigma_delta_alpha0",
        "sigma_delta_gamma0",
        "mu_a_sigma0",
        "tau_a_sigma0",
        "mu_l_sigma0",
        "tau_l_sigma0",
        "sigma_w_scale0",
        "sigma_w2_alpha0",
        "sigma_w2_gamma0",
        "rcs_coef_sigma0",
    ]
    for field in positive_float_fields:
        if float(getattr(args, field)) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} must be positive.")
    if not 0 < args.target_accept < 1:
        raise ValueError("--target-accept must be between 0 and 1.")
    if args.survival_model == "rcs":
        knots = np.asarray(args.rcs_knots, dtype=float)
        if knots.size < 4:
            raise ValueError("--rcs-knots must contain at least four knots.")
        if np.any(~np.isfinite(knots)) or np.any(np.diff(knots) <= 0):
            raise ValueError("--rcs-knots must be finite and strictly increasing.")
    if args.run_label is not None:
        args.run_label = safe_filename_token(args.run_label)
    args.exclude_centers = [str(center) for center in args.exclude_centers]
    return args


def safe_filename_token(value: str) -> str:
    """Convert a free-form label into a compact file-name token."""
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    token = re.sub(r"-{2,}", "-", token).strip("-_.")
    if not token:
        raise ValueError("--run-label must contain at least one file-name-safe character.")
    return token


def compact_float(value: float) -> str:
    """Format numeric settings for stable, readable file names."""
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def output_setting_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Return settings that can change the fitted posterior or analysis sample."""
    return {
        "survival_model": args.survival_model,
        "standardize_w": args.standardize_w,
        "min_center_n": int(args.min_center_n),
        "min_center_events": int(args.min_center_events),
        "min_center_nonevents": int(args.min_center_nonevents),
        "apply_center_filter": bool(args.apply_center_filter),
        "include_databases": list(args.include_databases),
        "exclude_centers": list(args.exclude_centers),
        "draws": int(args.draws),
        "tune": int(args.tune),
        "chains": int(args.chains),
        "cores": int(args.cores),
        "target_accept": float(args.target_accept),
        "betax_sigma0": float(args.betax_sigma0),
        "betaz_sigma0": float(args.betaz_sigma0),
        "beta0_first_mu0": float(args.beta0_first_mu0),
        "beta0_first_sigma0": float(args.beta0_first_sigma0),
        "sigma_delta_alpha0": float(args.sigma_delta_alpha0),
        "sigma_delta_gamma0": float(args.sigma_delta_gamma0),
        "mu_a_sigma0": float(args.mu_a_sigma0),
        "tau_a_sigma0": float(args.tau_a_sigma0),
        "mu_l_sigma0": float(args.mu_l_sigma0),
        "tau_l_sigma0": float(args.tau_l_sigma0),
        "sigma_w_prior": args.sigma_w_prior,
        "sigma_w_scale0": float(args.sigma_w_scale0),
        "sigma_w2_alpha0": float(args.sigma_w2_alpha0),
        "sigma_w2_gamma0": float(args.sigma_w2_gamma0),
        "rcs_knots": [float(x) for x in args.rcs_knots],
        "rcs_coef_sigma0": float(args.rcs_coef_sigma0),
    }


def output_setting_label(args: argparse.Namespace) -> str:
    """Build the suffix used to prevent different RFBBP settings from overwriting each other."""
    if args.run_label:
        return args.run_label

    payload = output_setting_payload(args)
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:10]
    readable = (
        f"{args.survival_model}"
        f"_d{args.draws}_t{args.tune}_c{args.chains}"
        f"_ta{compact_float(args.target_accept)}"
        f"_n{args.min_center_n}e{args.min_center_events}ne{args.min_center_nonevents}"
        f"_px{compact_float(args.betax_sigma0)}"
        f"_cal{compact_float(args.mu_a_sigma0)}-{compact_float(args.tau_a_sigma0)}"
        f"-{compact_float(args.mu_l_sigma0)}-{compact_float(args.tau_l_sigma0)}"
        f"_sw{args.sigma_w_prior}"
        f"_{digest}"
    )
    return safe_filename_token(readable)


def covariates_for(variable: str, adjustment: str) -> list[str]:
    """Return the covariate columns used by each formal RFBBP model."""
    if adjustment != "sofa_adjusted":
        raise ValueError("Formal RFBBP currently supports only sofa_adjusted datasets.")
    if variable not in SOFA_Z_COL:
        raise ValueError(f"Unknown RFBBP variable: {variable}")
    return [*BASIC_Z_COLS, SOFA_Z_COL[variable]]


def load_analysis_dataset(variable: str, adjustment: str, data_dir: Path) -> tuple[pd.DataFrame, Path]:
    """Load a prepared per-variable analysis dataset from the requested directory."""
    data_file = data_dir / f"{variable}_{adjustment}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing analysis dataset: {data_file}")
    return pd.read_csv(data_file), data_file


def filter_dataset(
    df: pd.DataFrame,
    z_cols: list[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply database, complete-case, and center-eligibility filters."""
    required_cols = ["database", "center_id", "exposure_analysis", "time", "event", *z_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    raw_n = int(len(df))
    raw_database_counts = {str(k): int(v) for k, v in df["database"].value_counts(dropna=False).items()}

    include_databases = tuple(args.include_databases)
    df = df[df["database"].isin(include_databases)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No rows remain after filtering database to {include_databases}.")

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    complete_meta = {
        "complete_case_n_rows": int(len(df)),
        "complete_case_n_centers": int(df["center_id"].nunique()),
        "complete_case_n_events": int(df["event"].sum()),
    }

    center_counts = df.groupby("center_id", observed=True).agg(
        n=("center_id", "size"),
        events=("event", "sum"),
    )
    center_counts["nonevents"] = center_counts["n"] - center_counts["events"]
    if args.apply_center_filter:
        eligible_centers = center_counts.index[
            (center_counts["n"] >= args.min_center_n)
            & (center_counts["events"] >= args.min_center_events)
            & (center_counts["nonevents"] >= args.min_center_nonevents)
        ]
    else:
        eligible_centers = center_counts.index
    df = df[df["center_id"].isin(eligible_centers)].reset_index(drop=True)
    excluded_centers_requested = [str(center) for center in getattr(args, "exclude_centers", [])]
    if excluded_centers_requested:
        present_before_exclusion = set(df["center_id"].astype(str).unique())
        excluded_centers_applied = sorted(set(excluded_centers_requested) & present_before_exclusion)
        df = df[~df["center_id"].astype(str).isin(excluded_centers_requested)].reset_index(drop=True)
    else:
        excluded_centers_applied = []
    if df["center_id"].nunique() < 2:
        raise ValueError("RFBBP requires at least two eligible centers.")

    metadata = {
        "raw_n_rows": raw_n,
        "raw_database_counts": raw_database_counts,
        **complete_meta,
        "n_rows": int(len(df)),
        "n_centers": int(df["center_id"].nunique()),
        "n_events": int(df["event"].sum()),
        "center_filter": {
            "disabled": not bool(args.apply_center_filter),
            "applied": bool(args.apply_center_filter),
            "min_n": int(args.min_center_n),
            "min_events": int(args.min_center_events),
            "min_nonevents": int(args.min_center_nonevents),
        },
        "excluded_centers": {
            "requested": excluded_centers_requested,
            "applied": excluded_centers_applied,
        },
        "include_databases": list(include_databases),
    }
    return df, metadata


def build_model_kwargs(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    """Collect model settings in one object so every variable uses the same RFBBP specification."""
    model_kwargs = {
        "cutpoints": CUTPOINTS,
        "auto_cutpoints": False,
        "study_event_cutpoint_constraint": False,
        "max_cuts": 15,
        "min_events_per_interval": 10,
        "hierarchical_calibration": True,
        "betax_sigma0": float(args.betax_sigma0),
        "betaz_sigma0": float(args.betaz_sigma0),
        "beta0_first_mu0": float(args.beta0_first_mu0),
        "beta0_first_sigma0": float(args.beta0_first_sigma0),
        "sigma_delta_alpha0": float(args.sigma_delta_alpha0),
        "sigma_delta_gamma0": float(args.sigma_delta_gamma0),
        "mu_a_sigma0": float(args.mu_a_sigma0),
        "tau_a_sigma0": float(args.tau_a_sigma0),
        "mu_l_sigma0": float(args.mu_l_sigma0),
        "tau_l_sigma0": float(args.tau_l_sigma0),
        "sigma_w_prior": args.sigma_w_prior,
        "sigma_w_scale0": float(args.sigma_w_scale0),
        "sigma_w2_alpha0": float(args.sigma_w2_alpha0),
        "sigma_w2_gamma0": float(args.sigma_w2_gamma0),
        "standardize_w": args.standardize_w,
        "solver": "pymc",
        "draws": int(args.draws),
        "tune": int(args.tune),
        "chains": int(args.chains),
        "mcmc_cores": int(args.cores),
        "target_accept": float(args.target_accept),
        "seed": int(seed),
        "pbar": True,
    }
    if args.survival_model == "rcs":
        model_kwargs.update(
            {
                "rcs_knots": [float(x) for x in args.rcs_knots],
                "rcs_coef_sigma0": float(args.rcs_coef_sigma0),
            }
        )
    return model_kwargs


def model_class_for(survival_model: str):
    """Return the RFBBP estimator class for the requested survival layer."""
    if survival_model == "pem":
        from rfbbp import RFBBPPyMC

        return RFBBPPyMC
    if survival_model == "cox":
        from rfbbp import RFBBPCox

        return RFBBPCox
    if survival_model == "rcs":
        from rfbbp import RFBBPRCS

        return RFBBPRCS
    raise ValueError(f"Unknown survival_model={survival_model!r}.")


def baseline_prior_label(survival_model: str) -> str:
    """Describe the baseline component used by the requested survival layer."""
    if survival_model == "pem":
        return "noncentered_random_walk"
    if survival_model in {"cox", "rcs"}:
        return "stratified_cox_partial_likelihood_unspecified_baseline"
    raise ValueError(f"Unknown survival_model={survival_model!r}.")


def run_one_variable(
    variable: str,
    adjustment: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run one formal RFBBP model and save all per-variable outputs."""
    data, data_file = load_analysis_dataset(variable, adjustment, args.data_dir)
    z_cols = covariates_for(variable, adjustment)
    data, metadata = filter_dataset(data, z_cols, args)

    s = data["center_id"].astype(str).to_numpy()
    w = data["exposure_analysis"].to_numpy(dtype=float)
    yt = data["time"].to_numpy(dtype=float)
    y1 = data["event"].to_numpy(dtype=int)
    z = data[z_cols].to_numpy(dtype=float)
    model_kwargs = build_model_kwargs(args, seed)
    estimator_class = model_class_for(args.survival_model)

    print("\n" + "=" * 80)
    print(f"RFBBP variable: {variable} ({adjustment})")
    print(f"Survival model: {args.survival_model}")
    print(f"Input data: {data_file}")
    print(f"Database filter: {', '.join(args.include_databases)}")
    print(
        "After filtering: "
        f"rows={len(data)}, centers={data['center_id'].nunique()}, events={int(y1.sum())}"
    )
    print(f"Exposure column: exposure_analysis; standardize_w={model_kwargs['standardize_w']}")
    print("Latent x parameterization: conditional")
    if args.survival_model == "pem":
        print("Baseline hazard prior: non-centered random-walk prior in RFBBPPyMC")
    elif args.survival_model == "cox":
        print("Baseline hazard: unspecified center-stratified Cox partial likelihood in RFBBPCox")
    else:
        print("Baseline hazard: unspecified center-stratified Cox partial likelihood with latent-X RCS in RFBBPRCS")
        print(f"RCS knots on latent X scale: {model_kwargs['rcs_knots']}")
    print(f"Output directory: {args.output_dir}")
    print("Starting RFBBP fit with progress bar...")

    model = estimator_class(**model_kwargs)
    model.fit(S=s, W=w, YT=yt, Y1=y1, Z=z)

    if args.survival_model == "rcs":
        summary_vars = (
            "gamma_x",
            "log_HR_x1_vs_x0",
            "HR_x1_vs_x0",
            "log_HR_x2_vs_x0",
            "HR_x2_vs_x0",
            "beta_z",
            "mu_a",
            "tau_a",
            "mu_l",
            "tau_l",
            "a",
            "b",
            "sigma_w",
        )
    else:
        summary_vars = (
            "beta_x",
            "HR_x",
            "beta_z",
            "mu_a",
            "tau_a",
            "mu_l",
            "tau_l",
            "a",
            "b",
            "sigma_w",
        )
    summary = model.summary(var_names=summary_vars)
    diagnostics = model.diagnostics()

    w_scale = w_label(args.standardize_w)
    setting_label = output_setting_label(args)
    model_file_token = {
        "pem": "rfbbp",
        "cox": "rfbbpcox",
        "rcs": "rfbbprcs",
    }[args.survival_model]
    database_label = "_".join(args.include_databases)
    prefix = f"{variable}_{adjustment}_{database_label}_w_{w_scale}_{model_file_token}_hierarchical_5interval_{setting_label}"
    summary_path = args.output_dir / f"{prefix}_summary.csv"
    diagnostics_path = args.output_dir / f"{prefix}_diagnostics.csv"
    idata_path = args.output_dir / f"{prefix}.nc"
    metadata_path = args.output_dir / f"{prefix}_metadata.json"
    curve_path = args.output_dir / f"{prefix}_hr_curve.csv"

    summary.to_csv(summary_path)
    diagnostics.to_csv(diagnostics_path)
    if args.skip_netcdf:
        idata_path = None
    else:
        model.idata_.to_netcdf(idata_path)
    if args.survival_model == "rcs":
        model.hazard_ratio_curve().to_csv(curve_path, index=False)

    metadata_path.write_text(
        json.dumps(
            {
                "variable": variable,
                "adjustment": adjustment,
                "survival_model": args.survival_model,
                "data_file": str(data_file),
                "include_databases": list(args.include_databases),
                "z_cols": z_cols,
                "model_kwargs": model_kwargs,
                "output_setting_label": setting_label,
                "output_setting_payload": output_setting_payload(args),
                "selected_cutpoints": (
                    [float(x) for x in np.asarray(model.cutpoints_).ravel()]
                    if getattr(model, "cutpoints_", None) is not None
                    else None
                ),
                **metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result_row = {
        "variable": variable,
        "adjustment": adjustment,
        "survival_model": args.survival_model,
        "status": "ok",
        "n": int(len(data)),
        "events": int(y1.sum()),
        "centers": int(data["center_id"].nunique()),
        "exposure": "exposure_analysis",
        "standardize_w": args.standardize_w,
        "w_label": w_scale,
        "output_setting_label": setting_label,
        "baseline_prior": baseline_prior_label(args.survival_model),
        "summary_csv": str(summary_path),
        "diagnostics_csv": str(diagnostics_path),
        "metadata_json": str(metadata_path),
        "netcdf": str(idata_path) if idata_path is not None else "",
    }
    if args.survival_model == "rcs":
        log_hr1 = summary.loc["log_HR_x1_vs_x0"]
        hr1 = summary.loc["HR_x1_vs_x0"]
        log_hr2 = summary.loc["log_HR_x2_vs_x0"]
        hr2 = summary.loc["HR_x2_vs_x0"]
        result_row.update(
            {
                "beta_x_mean": np.nan,
                "beta_x_post_sd": np.nan,
                "beta_x_cri_low": np.nan,
                "beta_x_cri_high": np.nan,
                "HR_x_mean": np.nan,
                "HR_x_post_sd": np.nan,
                "HR_x_cri_low": np.nan,
                "HR_x_cri_high": np.nan,
                "log_HR_x1_vs_x0_mean": float(log_hr1["estimate"]),
                "log_HR_x1_vs_x0_post_sd": float(log_hr1["post_sd"]),
                "log_HR_x1_vs_x0_cri_low": float(log_hr1["CI_1"]),
                "log_HR_x1_vs_x0_cri_high": float(log_hr1["CI_2"]),
                "HR_x1_vs_x0_mean": float(hr1["estimate"]),
                "HR_x1_vs_x0_post_sd": float(hr1["post_sd"]),
                "HR_x1_vs_x0_cri_low": float(hr1["CI_1"]),
                "HR_x1_vs_x0_cri_high": float(hr1["CI_2"]),
                "log_HR_x2_vs_x0_mean": float(log_hr2["estimate"]),
                "log_HR_x2_vs_x0_post_sd": float(log_hr2["post_sd"]),
                "log_HR_x2_vs_x0_cri_low": float(log_hr2["CI_1"]),
                "log_HR_x2_vs_x0_cri_high": float(log_hr2["CI_2"]),
                "HR_x2_vs_x0_mean": float(hr2["estimate"]),
                "HR_x2_vs_x0_post_sd": float(hr2["post_sd"]),
                "HR_x2_vs_x0_cri_low": float(hr2["CI_1"]),
                "HR_x2_vs_x0_cri_high": float(hr2["CI_2"]),
                "hr_curve_csv": str(curve_path),
            }
        )
    else:
        beta = summary.loc["beta_x"]
        hr = summary.loc["HR_x"]
        result_row.update(
            {
                "beta_x_mean": float(beta["estimate"]),
                "beta_x_post_sd": float(beta["post_sd"]),
                "beta_x_cri_low": float(beta["CI_1"]),
                "beta_x_cri_high": float(beta["CI_2"]),
                "HR_x_mean": float(hr["estimate"]),
                "HR_x_post_sd": float(hr["post_sd"]),
                "HR_x_cri_low": float(hr["CI_1"]),
                "HR_x_cri_high": float(hr["CI_2"]),
            }
        )
    diagnostic_row = {
        "variable": variable,
        "adjustment": adjustment,
        "survival_model": args.survival_model,
        "status": "ok",
        "standardize_w": args.standardize_w,
        "w_label": w_scale,
        "output_setting_label": setting_label,
        **{str(k): float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in diagnostics.to_dict().items()},
        "summary_csv": str(summary_path),
        "diagnostics_csv": str(diagnostics_path),
        "metadata_json": str(metadata_path),
        "netcdf": str(idata_path) if idata_path is not None else "",
    }

    print("\nPosterior summary:")
    print(summary)
    print("\nDiagnostics:")
    print(diagnostics)
    print("\nSaved files:")
    print(summary_path)
    print(diagnostics_path)
    if idata_path is not None:
        print(idata_path)
    else:
        print("(NetCDF posterior skipped)")
    print(metadata_path)
    if args.survival_model == "rcs":
        print(curve_path)
    return result_row, diagnostic_row


def main() -> None:
    set_runtime_env()
    args = normalize_args(parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    table_dir = args.table_dir
    table_dir.mkdir(parents=True, exist_ok=True)
    setting_label = output_setting_label(args)
    aggregate_prefix = {
        "pem": "rfbbp",
        "cox": "rfbbpcox",
        "rcs": "rfbbprcs",
    }[args.survival_model]
    if args.standardize_w == "none":
        aggregate_results_path = table_dir / f"{aggregate_prefix}_results_{setting_label}.csv"
        aggregate_diagnostics_path = table_dir / f"{aggregate_prefix}_diagnostics_{setting_label}.csv"
    else:
        aggregate_results_path = table_dir / f"{aggregate_prefix}_results_w_{w_label(args.standardize_w)}_{setting_label}.csv"
        aggregate_diagnostics_path = table_dir / f"{aggregate_prefix}_diagnostics_w_{w_label(args.standardize_w)}_{setting_label}.csv"

    result_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    print("Formal RFBBP batch settings:")
    print(f"  variables: {', '.join(args.variables)}")
    print(f"  adjustment: {args.adjustment}")
    print(f"  survival_model: {args.survival_model}")
    print(f"  standardize_w: {args.standardize_w}")
    print(f"  include_databases: {', '.join(args.include_databases)}")
    if args.apply_center_filter:
        print(
            "  center filter: "
            f"n>={args.min_center_n}, events>={args.min_center_events}, nonevents>={args.min_center_nonevents}"
        )
    else:
        print("  center filter: disabled after complete-case filtering")
    print("  x_parameterization: conditional")
    print(f"  draws/tune/chains/cores: {args.draws}/{args.tune}/{args.chains}/{args.cores}")
    print(f"  target_accept: {args.target_accept}")
    print(f"  output_setting_label: {setting_label}")
    print(f"  per-variable output_dir: {args.output_dir}")
    print(f"  aggregate_results: {aggregate_results_path}")
    print(f"  aggregate_diagnostics: {aggregate_diagnostics_path}")

    for i, variable in enumerate(args.variables):
        try:
            result_row, diagnostic_row = run_one_variable(
                variable=variable,
                adjustment=args.adjustment,
                args=args,
                seed=int(args.seed + i),
            )
            if result_row is not None:
                result_rows.append(result_row)
            diagnostic_rows.append(diagnostic_row)
        except KeyboardInterrupt:
            print("\nInterrupted by user; partial aggregate files will be saved.")
            break
        except Exception as exc:
            diagnostic_rows.append({
                "variable": variable,
                "adjustment": args.adjustment,
                "survival_model": args.survival_model,
                "status": "failed",
                "error": repr(exc),
            })
            print(f"\n[WARN] RFBBP failed for {variable}: {exc!r}")

        pd.DataFrame(result_rows).to_csv(aggregate_results_path, index=False)
        pd.DataFrame(diagnostic_rows).to_csv(aggregate_diagnostics_path, index=False)
        print(f"\nAggregate results updated: {aggregate_results_path}")
        print(f"Aggregate diagnostics updated: {aggregate_diagnostics_path}")

    print("\nFinished RFBBP batch.")
    print(f"Successful models: {len(result_rows)} / {len(args.variables)}")
    print(f"Aggregate results: {aggregate_results_path}")
    print(f"Aggregate diagnostics: {aggregate_diagnostics_path}")


if __name__ == "__main__":
    main()
