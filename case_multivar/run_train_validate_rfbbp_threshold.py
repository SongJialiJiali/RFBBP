#!/usr/bin/env python3
"""Train-only RFBBP held-out threshold transportability for three biomarkers.

This module intentionally contains its threshold-analysis helpers so the
workflow can be maintained without synchronizing a separate utility script.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

for name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
]:
    os.environ.setdefault(name, "1")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/rfbbp-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rfbbp-matplotlib")
os.environ.setdefault(
    "PYTENSOR_FLAGS",
    "base_compiledir=/tmp/rfbbp-pytensor,compiledir=/tmp/rfbbp-pytensor/compiledir",
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.duration.hazard_regression import PHReg

from rfbbp import RFBBPPyMC


SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_FINAL_DIR = SCRIPT_DIR / "analysis_final"
DATA_DIR = ANALYSIS_FINAL_DIR / "analysis_datasets"
OUTPUT_DIR = ANALYSIS_FINAL_DIR / "RFBBP_result"
# Final-analysis tables are intentionally written directly under RFBBP_result:
# rfbbp_case_final.ipynb reads the split and calibration-parameter files there.
TABLE_DIR = OUTPUT_DIR
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "train_validate_models"

OUT_PREFIX = "three_biomarkers_train70_validate30_rfbbp_threshold_transportability"
OUT_SCAN = TABLE_DIR / f"{OUT_PREFIX}_scan.csv"
OUT_CENTER_HR = TABLE_DIR / f"{OUT_PREFIX}_center_threshold_hr.csv"
OUT_SPLIT = TABLE_DIR / f"{OUT_PREFIX}_split_assignment.csv"
RUN_TAG = "train70_validate30"

CUTPOINTS = [0.0, 1.0, 3.0, 7.0, 14.0, 28.0]

MIN_SLOPE_N = 20
MIN_SLOPE_EVENTS = 5
MIN_SLOPE_NONEVENTS = 5

BLUE = "#2F78B7"
ORANGE = "#D95F02"
PURPLE = "#6A51A3"
GREY = "#9AA3AD"
LIGHT_GREY = "#D9DEE5"
TEXT = "#222222"
GRID = "#E4E8ED"


@dataclass(frozen=True)
class BiomarkerSpec:
    key: str
    label: str
    unit: str
    data_file: Path
    summary_file: Path
    sofa_col: str
    references: tuple[float, ...]
    direction: str
    raw_label: str


HEMOGLOBIN_SPEC = BiomarkerSpec(
    key="hemoglobin",
    label="Hemoglobin",
    unit="g/dL",
    data_file=DATA_DIR / "Hemoglobin_hour_24_sofa_adjusted.csv",
    summary_file=Path(""),
    sofa_col="SOFA_score_hour_24_z",
    references=(8.0, 10.0, 12.0),
    direction="low",
    raw_label="low hemoglobin",
)


CREATININE_SPEC = BiomarkerSpec(
    key="creatinine",
    label="Creatinine",
    unit="mg/dL",
    data_file=DATA_DIR / "Creatinine_hour_24_sofa_adjusted.csv",
    summary_file=Path(""),
    sofa_col="SOFA_no_renal_z",
    references=(1.0, 2.0, 3.0),
    direction="high",
    raw_label="high creatinine",
)

LACTATE_SPEC = BiomarkerSpec(
    key="lactate",
    label="Lactate",
    unit="mmol/L",
    data_file=DATA_DIR / "Lactate_hour_24_sofa_adjusted.csv",
    summary_file=Path(""),
    sofa_col="SOFA_score_hour_24_z",
    references=(2.0, 3.5, 4.0, 8.0),
    direction="high",
    raw_label="high lactate",
)

ALL_SPECS = (HEMOGLOBIN_SPEC, LACTATE_SPEC, CREATININE_SPEC)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
        }
    )


def analysis_value(spec: BiomarkerSpec, value: float) -> float:
    return float(np.log(value)) if spec.key in {"creatinine", "lactate"} else float(value)


def raw_value(
    spec: BiomarkerSpec,
    value: pd.Series | np.ndarray | float,
) -> pd.Series | np.ndarray | float:
    return np.exp(value) if spec.key in {"creatinine", "lactate"} else value


def load_data(spec: BiomarkerSpec) -> pd.DataFrame:
    cols = [
        "stay_id", "database", "center_id", "time", "event",
        "exposure_analysis", "age_z", "gender_male", spec.sofa_col,
    ]
    data = pd.read_csv(spec.data_file, usecols=cols)
    data = data.loc[data["database"].isin(["mimic_iv", "eicu"])].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=cols).copy()
    data = data.loc[data["time"].gt(0) & data["event"].isin([0, 1])].copy()
    data["center_id"] = data["center_id"].astype(str)
    data["database"] = data["database"].astype(str)
    data["event"] = data["event"].astype(int)
    return data.reset_index(drop=True)


def indexed_parameter(summary: pd.DataFrame, name: str) -> pd.Series:
    pattern = re.compile(rf"^{re.escape(name)}\[(\d+)\]$")
    values: dict[int, float] = {}
    for index, row in summary.iterrows():
        match = pattern.match(str(index))
        if match:
            values[int(match.group(1))] = float(row["estimate"])
    return pd.Series(values, dtype=float).sort_index()


def build_equivalent_thresholds(
    spec: BiomarkerSpec,
    params: pd.DataFrame,
    reference: float,
    reference_strategy: str = "nearest_center",
) -> pd.DataFrame:
    out = params.copy()
    ref_analysis = analysis_value(spec, reference)
    out["x_ref_if_reference"] = (ref_analysis - out["a"]) / out["b"]
    median_state = float(out["x_ref_if_reference"].median())
    reference_idx = int((out["x_ref_if_reference"] - median_state).abs().idxmin())
    reference_row = out.loc[reference_idx]
    if reference_strategy == "median":
        x_ref = median_state
        reference_center_id = "cross-centre median"
    elif reference_strategy == "nearest_center":
        x_ref = float(reference_row["x_ref_if_reference"])
        reference_center_id = str(reference_row["center_id"])
    else:
        raise ValueError(f"Unknown reference strategy: {reference_strategy}")
    out["reference_value"] = float(reference)
    out["reference_center_id"] = reference_center_id
    out["reference_x_state"] = x_ref
    out["equivalent_analysis_threshold"] = out["a"] + out["b"] * x_ref
    out["equivalent_raw_threshold"] = raw_value(spec, out["equivalent_analysis_threshold"])
    return out


def classify_for_threshold(
    spec: BiomarkerSpec,
    validation: pd.DataFrame,
    thresholds: pd.DataFrame,
    reference: float,
) -> pd.DataFrame:
    out = validation.merge(
        thresholds[["center_id", "equivalent_analysis_threshold", "equivalent_raw_threshold"]],
        on="center_id",
        how="inner",
    ).copy()
    ref_analysis = analysis_value(spec, reference)
    if spec.direction == "low":
        out["raw_abnormal"] = out["exposure_analysis"].le(ref_analysis)
        out["rfbbp_abnormal"] = out["exposure_analysis"].le(out["equivalent_analysis_threshold"])
    else:
        out["raw_abnormal"] = out["exposure_analysis"].ge(ref_analysis)
        out["rfbbp_abnormal"] = out["exposure_analysis"].ge(out["equivalent_analysis_threshold"])
    out["reference_value"] = float(reference)
    out["classification_changed"] = out["raw_abnormal"].ne(out["rfbbp_abnormal"])
    return out


def fit_center_binary_threshold(
    group: pd.DataFrame,
    exposure_col: str,
    sofa_col: str,
) -> dict[str, object]:
    n = int(len(group))
    events = int(group["event"].sum())
    nonevents = n - events
    abnormal = group[exposure_col].astype(bool)
    abnormal_n = int(abnormal.sum())
    normal_n = n - abnormal_n
    abnormal_events = int(group.loc[abnormal, "event"].sum())
    normal_events = int(group.loc[~abnormal, "event"].sum())
    base = {
        "n": n, "events": events, "nonevents": nonevents,
        "abnormal_n": abnormal_n, "normal_n": normal_n,
        "abnormal_events": abnormal_events, "normal_events": normal_events,
    }
    if (
        n < MIN_SLOPE_N
        or events < MIN_SLOPE_EVENTS
        or nonevents < MIN_SLOPE_NONEVENTS
        or abnormal_n < 5
        or normal_n < 5
        or abnormal_events < 1
        or normal_events < 1
    ):
        return {**base, "status": "skipped_sparse_threshold_groups"}

    covariates = ["age_z", "gender_male", sofa_col]
    use_cols = [exposure_col, *[c for c in covariates if group[c].nunique(dropna=True) > 1]]
    model_data = group.copy()
    model_data[exposure_col] = model_data[exposure_col].astype(float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = PHReg(
                model_data["time"].to_numpy(float),
                model_data[use_cols].to_numpy(float),
                status=model_data["event"].to_numpy(int),
                ties="efron",
            ).fit(disp=False, maxiter=100)
        beta = float(fit.params[0])
        se = float(fit.bse[0])
        if not np.isfinite(beta) or not np.isfinite(se) or se <= 0:
            return {**base, "status": "failed_nonfinite"}
        return {
            **base,
            "status": "ok",
            "log_hr": beta,
            "se": se,
            "hr": float(np.exp(beta)),
            "hr_ci_low": float(np.exp(beta - 1.96 * se)),
            "hr_ci_high": float(np.exp(beta + 1.96 * se)),
        }
    except Exception as exc:
        return {**base, "status": "failed", "error": repr(exc)}


def estimate_meta(slopes: pd.DataFrame, exposure_label: str) -> dict[str, object]:
    ok = slopes.loc[slopes["status"].eq("ok")].copy()
    if ok.empty:
        return {
            "exposure": exposure_label,
            "centres_fitted": 0,
            "tau2": np.nan,
            "I2_percent": np.nan,
            "pooled_hr": np.nan,
        }
    y = ok["log_hr"].to_numpy(float)
    se = ok["se"].to_numpy(float)
    variance = se**2
    weights = 1.0 / variance
    fixed = float(np.sum(weights * y) / np.sum(weights))
    q = float(np.sum(weights * (y - fixed) ** 2))
    k = len(y)
    c = float(np.sum(weights) - np.sum(weights**2) / np.sum(weights))
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 and c > 0 else 0.0
    random_weights = 1.0 / (variance + tau2)
    pooled = float(np.sum(random_weights * y) / np.sum(random_weights))
    i2 = max(0.0, (q - (k - 1)) / q) * 100.0 if q > 0 and k > 1 else 0.0
    return {
        "exposure": exposure_label,
        "centres_fitted": int(k),
        "tau2": tau2,
        "I2_percent": i2,
        "pooled_hr": float(np.exp(pooled)),
    }


def run_center_threshold_hrs(spec: BiomarkerSpec, classified: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    threshold_specs = [
        ("raw_abnormal", "Uniform threshold"),
        ("rfbbp_abnormal", "RFBBP-equivalent threshold"),
    ]
    for center_id, group in classified.groupby("center_id", sort=False):
        database = str(group["database"].iloc[0])
        reference = float(group["reference_value"].iloc[0])
        equivalent_threshold = float(group["equivalent_raw_threshold"].iloc[0])
        for column, label in threshold_specs:
            row = fit_center_binary_threshold(group, column, spec.sofa_col)
            row.update(
                {
                    "biomarker": spec.label,
                    "center_id": center_id,
                    "database": database,
                    "reference_value": reference,
                    "equivalent_raw_threshold": equivalent_threshold,
                    "threshold": label,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def common_threshold_meta(center_hr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Index]:
    ok = center_hr.loc[center_hr["status"].eq("ok")].copy()
    raw = ok.loc[ok["threshold"].eq("Uniform threshold")].set_index("center_id")
    calibrated = ok.loc[ok["threshold"].eq("RFBBP-equivalent threshold")].set_index("center_id")
    common_centres = raw.index.intersection(calibrated.index)
    meta = pd.DataFrame(
        [
            estimate_meta(raw.loc[common_centres].reset_index(), "Uniform threshold"),
            estimate_meta(
                calibrated.loc[common_centres].reset_index(),
                "RFBBP-equivalent threshold",
            ),
        ]
    )
    return meta, common_centres


def format_meta(meta: pd.DataFrame, threshold: str) -> str:
    row = meta.loc[meta["exposure"].eq(threshold)].iloc[0]
    return rf"$\tau^2$={float(row['tau2']):.3f}; $I^2$={float(row['I2_percent']):.1f}%"


def plot_reference_figure(
    spec: BiomarkerSpec,
    reference: float,
    classified: pd.DataFrame,
    scan_row: pd.Series,
    center_hr: pd.DataFrame,
    output_suffix: str = "full_rfbbp_exploratory",
) -> None:
    thresholds = (
        classified[["center_id", "database", "equivalent_raw_threshold"]]
        .drop_duplicates()
        .sort_values("equivalent_raw_threshold")
        .reset_index(drop=True)
    )
    reclassification = (
        classified.groupby(["center_id", "database"], as_index=False)
        .agg(validation_n=("event", "size"), changed_n=("classification_changed", "sum"))
    )
    reclassification["changed_percent"] = (
        100 * reclassification["changed_n"] / reclassification["validation_n"]
    )
    threshold_meta, common_centres = common_threshold_meta(center_hr)

    fig = plt.figure(figsize=(12.2, 7.3), dpi=450)
    grid = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.05], width_ratios=[1, 1], hspace=0.46, wspace=0.34
    )
    ax_threshold = fig.add_subplot(grid[0, 0])
    ax_reclassification = fig.add_subplot(grid[0, 1])
    ax_hr = fig.add_subplot(grid[1, :])

    y = np.arange(len(thresholds))
    is_mimic = thresholds["database"].eq("mimic_iv").to_numpy()
    ax_threshold.scatter(
        thresholds.loc[~is_mimic, "equivalent_raw_threshold"],
        y[~is_mimic],
        s=16,
        color=BLUE,
        alpha=0.72,
        linewidth=0,
        label="eICU centres",
    )
    ax_threshold.scatter(
        thresholds.loc[is_mimic, "equivalent_raw_threshold"],
        y[is_mimic],
        s=42,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.6,
        label="MIMIC-IV centre",
        zorder=4,
    )
    ax_threshold.axvline(reference, color="#555555", linestyle=(0, (4, 4)), linewidth=1.1)
    quantiles = thresholds["equivalent_raw_threshold"].quantile([0.25, 0.5, 0.75])
    ax_threshold.axvspan(
        float(quantiles.loc[0.25]),
        float(quantiles.loc[0.75]),
        color=LIGHT_GREY,
        alpha=0.5,
        zorder=0,
    )
    ax_threshold.text(
        reference,
        len(thresholds) * 0.05,
        f"Uniform clinical threshold: {reference:g} {spec.unit}",
        rotation=90,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    ax_threshold.text(
        0.98,
        0.04,
        f"Median {quantiles.loc[0.5]:.1f} {spec.unit}\n"
        f"IQR {quantiles.loc[0.25]:.1f}-{quantiles.loc[0.75]:.1f} {spec.unit}\n"
        f"Range {scan_row['equivalent_threshold_min']:.1f}-"
        f"{scan_row['equivalent_threshold_max']:.1f} {spec.unit}",
        transform=ax_threshold.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=TEXT,
    )
    ax_threshold.set_xlabel(f"Equivalent local {spec.label.lower()} threshold ({spec.unit})")
    ax_threshold.set_ylabel("Centres ranked by equivalent threshold")
    ax_threshold.set_yticks([])
    ax_threshold.set_title(
        f"A. Centre-specific {spec.label.lower()} values equivalent to a common\n"
        f"{reference:g} {spec.unit} state",
        loc="left",
        fontweight="bold",
    )
    ax_threshold.grid(axis="x", color=GRID, linewidth=0.7)
    ax_threshold.spines[["top", "right"]].set_visible(False)
    ax_threshold.legend(frameon=False, loc="upper left", fontsize=8)

    reclassification = reclassification.sort_values("changed_percent").reset_index(drop=True)
    y_reclassification = np.arange(len(reclassification))
    is_mimic = reclassification["database"].eq("mimic_iv").to_numpy()
    zero_mask = reclassification["changed_percent"].eq(0).to_numpy()
    x_reclassification = reclassification["changed_percent"].to_numpy(float).copy()
    if zero_mask.any():
        x_reclassification[zero_mask] = np.linspace(-0.03, 0.03, int(zero_mask.sum()))
    ax_reclassification.scatter(
        x_reclassification[~is_mimic], y_reclassification[~is_mimic],
        s=16, color=BLUE, alpha=0.72, linewidth=0,
    )
    ax_reclassification.scatter(
        x_reclassification[is_mimic], y_reclassification[is_mimic],
        s=42, color=ORANGE, edgecolor="white", linewidth=0.6, zorder=4,
    )
    overall_changed = (
        reclassification["changed_n"].sum() / reclassification["validation_n"].sum() * 100
    )
    center_quantiles = reclassification["changed_percent"].quantile([0.25, 0.5, 0.75])
    ax_reclassification.axvline(overall_changed, color=PURPLE, linewidth=1.4)
    ax_reclassification.text(
        overall_changed + 0.15,
        len(reclassification) * 0.05,
        f"Overall changed: {overall_changed:.1f}%",
        rotation=90,
        ha="left",
        va="bottom",
        fontsize=8,
        color=PURPLE,
    )
    if zero_mask.any():
        ax_reclassification.text(
            0.01,
            float(y_reclassification[zero_mask].max()) + 2.0,
            f"0% reclassification: n={int(zero_mask.sum())} centres",
            ha="left",
            va="bottom",
            fontsize=8,
            color="#4A4A4A",
        )
    ax_reclassification.text(
        0.98,
        0.04,
        f"Overall: {overall_changed:.1f}% of patients\n"
        f"Median centre: {center_quantiles.loc[0.5]:.1f}% "
        f"(IQR {center_quantiles.loc[0.25]:.1f}-{center_quantiles.loc[0.75]:.1f}%)",
        transform=ax_reclassification.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=TEXT,
    )
    ax_reclassification.set_xlabel("Validation-subset patients reclassified (%)")
    ax_reclassification.set_ylabel("Centres ranked by reclassification")
    ax_reclassification.set_yticks([])
    ax_reclassification.set_title(
        "B. Reclassification using centre-specific\nequivalent thresholds",
        loc="left",
        fontweight="bold",
    )
    ax_reclassification.grid(axis="x", color=GRID, linewidth=0.7)
    ax_reclassification.spines[["top", "right"]].set_visible(False)

    ok = center_hr.loc[center_hr["status"].eq("ok")].copy()
    raw = ok.loc[ok["threshold"].eq("Uniform threshold")].set_index("center_id")
    calibrated = ok.loc[
        ok["threshold"].eq("RFBBP-equivalent threshold")
    ].set_index("center_id")
    paired = pd.DataFrame(
        {
            "raw_log_hr": raw.loc[common_centres, "log_hr"],
            "calibrated_log_hr": calibrated.loc[common_centres, "log_hr"],
        }
    ).sort_values("raw_log_hr")
    x = np.arange(len(paired))
    for x_value, (_, row) in zip(x, paired.iterrows()):
        ax_hr.plot(
            [x_value - 0.12, x_value + 0.12],
            [row["raw_log_hr"], row["calibrated_log_hr"]],
            color="#DCE1E7",
            linewidth=0.55,
            zorder=1,
        )
    ax_hr.scatter(x - 0.12, paired["raw_log_hr"], color=GREY, s=15, alpha=0.5, zorder=2)
    ax_hr.scatter(
        x + 0.12, paired["calibrated_log_hr"], color=PURPLE, s=15, alpha=0.82, zorder=3
    )
    ax_hr.axhline(0, color="#555555", linestyle=(0, (4, 4)), linewidth=0.8)
    ax_hr.set_xticks([])
    ax_hr.set_xlabel("Centres with estimable validation-subset threshold contrast")
    ax_hr.set_ylabel(f"Adjusted log HR, {spec.raw_label} vs reference")
    ax_hr.set_title(
        "C. Prognostic contrasts using uniform and centre-specific equivalent thresholds",
        loc="left",
        fontweight="bold",
    )
    ax_hr.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    ax_hr.spines[["top", "right"]].set_visible(False)
    ax_hr.legend(
        handles=[
            plt.Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=GREY, markersize=5,
                label=(
                    f"Uniform {reference:g} {spec.unit} "
                    f"({format_meta(threshold_meta, 'Uniform threshold')})"
                ),
            ),
            plt.Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=PURPLE, markersize=5,
                label=(
                    "RFBBP-equivalent threshold "
                    f"({format_meta(threshold_meta, 'RFBBP-equivalent threshold')})"
                ),
            ),
        ],
        frameon=False,
        loc="upper left",
        fontsize=8.5,
    )

    fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.11)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{spec.key}_threshold_transportability_reference{reference:g}_{output_suffix}".replace(
        ".", "p"
    )
    output_figure = FIGURE_DIR / f"{stem}.png"
    fig.savefig(output_figure, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(output_figure.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def split_within_center(data: pd.DataFrame, train_frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in data.groupby("center_id", sort=False):
        group = group.copy()
        order = rng.permutation(len(group))
        n_train = int(np.floor(train_frac * len(group)))
        n_train = min(max(n_train, 1), len(group) - 1) if len(group) > 1 else len(group)
        split = np.full(len(group), "validation", dtype=object)
        split[order[:n_train]] = "train"
        group["split"] = split
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--biomarker",
        choices=["hemoglobin", "lactate", "creatinine", "all"],
        default="all",
        help="Biomarker to run.",
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--run-tag", default=None, help="Output tag; defaults to trainXX_validateYY.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the three *_sofa_adjusted.csv analysis datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=(
            "Artifact root. Final CSV tables are written directly here, model files under "
            "train_validate_models/, and figures under figures/."
        ),
    )
    parser.add_argument("--reference-strategy", choices=["nearest_center", "median"], default="nearest_center")
    parser.add_argument(
        "--references",
        type=float,
        nargs="+",
        default=None,
        help="Override the clinical reference thresholds for the selected biomarker(s).",
    )
    parser.add_argument("--split-only", action="store_true", help="Save the patient split without fitting RFBBP.")
    parser.add_argument("--reference-set", choices=["legacy", "final"], default="legacy",
                        help="Use legacy thresholds or final thresholds (Hb 8/12, lactate 4/8, creatinine 1.5/2.5).")
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def output_paths(spec: BiomarkerSpec) -> dict[str, Path]:
    prefix = f"{spec.key}_{RUN_TAG}_rfbbp_threshold_transportability"
    return {
        "summary": MODEL_DIR / f"{prefix}_rfbbp_summary.csv",
        "diagnostics": MODEL_DIR / f"{prefix}_rfbbp_diagnostics.csv",
        "idata": MODEL_DIR / f"{prefix}.nc",
        "params": TABLE_DIR / f"{prefix}_training_calibration_parameters.csv",
        "metadata": MODEL_DIR / f"{prefix}_metadata.json",
    }


def fit_training_rfbbp(
    spec: BiomarkerSpec,
    train: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    paths = output_paths(spec)
    if args.reuse_existing and paths["summary"].exists() and paths["params"].exists():
        return pd.read_csv(paths["params"])

    z = train[["age_z", "gender_male", spec.sofa_col]].to_numpy(float)
    model = RFBBPPyMC(
        cutpoints=CUTPOINTS,
        auto_cutpoints=False,
        study_event_cutpoint_constraint=False,
        max_cuts=15,
        min_events_per_interval=10,
        hierarchical_calibration=True,
        betax_sigma0=2.5,
        betaz_sigma0=1.0,
        beta0_first_mu0=0.0,
        beta0_first_sigma0=1.0,
        sigma_delta_alpha0=3.0,
        sigma_delta_gamma0=0.5,
        mu_a_sigma0=0.5,
        tau_a_sigma0=0.5,
        mu_l_sigma0=0.5,
        tau_l_sigma0=0.5,
        sigma_w_prior="invgamma",
        sigma_w2_alpha0=3.0,
        sigma_w2_gamma0=1.0,
        standardize_w="none",
        solver="pymc",
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        mcmc_cores=min(args.cores, args.chains),
        target_accept=args.target_accept,
        seed=args.seed,
        pbar=True,
    )
    model.fit(
        S=train["center_id"].to_numpy(str),
        W=train["exposure_analysis"].to_numpy(float),
        YT=train["time"].to_numpy(float),
        Y1=train["event"].to_numpy(int),
        Z=z,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    summary = model.summary(
        var_names=("beta_x", "HR_x", "beta_z", "mu_a", "tau_a", "mu_l", "tau_l", "a", "b", "sigma_w")
    )
    diagnostics = model.diagnostics()
    summary.to_csv(paths["summary"])
    diagnostics.to_csv(paths["diagnostics"])
    model.idata_.to_netcdf(paths["idata"])

    centres = np.asarray(model.studies_, dtype=str)
    database_by_centre = train.drop_duplicates("center_id").set_index("center_id")["database"]
    a = indexed_parameter(summary, "a")
    b = indexed_parameter(summary, "b")
    sigma_w = indexed_parameter(summary, "sigma_w")
    if not (len(centres) == len(a) == len(b) == len(sigma_w)):
        raise ValueError(f"{spec.label}: RFBBP parameter count does not match training centre count.")

    params = pd.DataFrame(
        {
            "center_index": np.arange(len(centres)),
            "center_id": centres,
            "database": [database_by_centre.loc[c] for c in centres],
            "a": a.to_numpy(float),
            "b": b.to_numpy(float),
            "sigma_w": sigma_w.to_numpy(float),
            "beta_x": float(summary.loc["beta_x", "estimate"]),
        }
    )
    params.to_csv(paths["params"], index=False)
    return params


def run_one_spec(spec: BiomarkerSpec, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = output_paths(spec)
    data = load_data(spec)
    split = split_within_center(data, args.train_frac, args.seed)
    train = split.loc[split["split"].eq("train")].copy()
    validation = split.loc[split["split"].eq("validation")].copy()
    params = fit_training_rfbbp(spec, train, args)

    scan_rows: list[dict[str, object]] = []
    center_frames: list[pd.DataFrame] = []
    split_out = split[["stay_id", "database", "center_id", "time", "event", "exposure_analysis", "split"]].copy()
    split_out.insert(0, "biomarker", spec.label)

    for reference in spec.references:
        thresholds = build_equivalent_thresholds(spec, params, reference, reference_strategy=args.reference_strategy)
        classified = classify_for_threshold(spec, validation, thresholds, reference)
        classified_path = TABLE_DIR / (
            f"{spec.key}_{RUN_TAG}_rfbbp_threshold_transportability_"
            f"reference{reference:g}_validation_classification.csv".replace(".", "p")
        )
        thresholds_path = TABLE_DIR / (
            f"{spec.key}_{RUN_TAG}_rfbbp_threshold_transportability_"
            f"reference{reference:g}_equivalent_thresholds.csv".replace(".", "p")
        )
        classified.to_csv(classified_path, index=False)
        thresholds.to_csv(thresholds_path, index=False)

        center_hr = run_center_threshold_hrs(spec, classified)
        center_frames.append(center_hr)
        common_meta, common_centres = common_threshold_meta(center_hr)
        raw_meta = common_meta.loc[common_meta["exposure"].eq("Uniform threshold")].iloc[0]
        cal_meta = common_meta.loc[common_meta["exposure"].eq("RFBBP-equivalent threshold")].iloc[0]
        scan_row = {
            "biomarker": spec.label,
            "reference_value": float(reference),
            "unit": spec.unit,
            "direction": spec.direction,
            "reference_center_id": thresholds["reference_center_id"].iloc[0],
            "equivalent_threshold_min": thresholds["equivalent_raw_threshold"].min(),
            "equivalent_threshold_p25": thresholds["equivalent_raw_threshold"].quantile(0.25),
            "equivalent_threshold_median": thresholds["equivalent_raw_threshold"].median(),
            "equivalent_threshold_p75": thresholds["equivalent_raw_threshold"].quantile(0.75),
            "equivalent_threshold_max": thresholds["equivalent_raw_threshold"].max(),
            "validation_n": int(len(classified)),
            "reclassified_n": int(classified["classification_changed"].sum()),
            "reclassified_percent": float(classified["classification_changed"].mean() * 100),
            "raw_centres_fitted": raw_meta["centres_fitted"],
            "calibrated_centres_fitted": cal_meta["centres_fitted"],
            "common_centres_fitted": len(common_centres),
            "raw_tau2": raw_meta["tau2"],
            "calibrated_tau2": cal_meta["tau2"],
            "tau2_absolute_change": raw_meta["tau2"] - cal_meta["tau2"],
            "raw_I2_percent": raw_meta["I2_percent"],
            "calibrated_I2_percent": cal_meta["I2_percent"],
            "I2_absolute_change": raw_meta["I2_percent"] - cal_meta["I2_percent"],
            "raw_pooled_hr": raw_meta["pooled_hr"],
            "calibrated_pooled_hr": cal_meta["pooled_hr"],
        }
        scan_rows.append(scan_row)
        plot_reference_figure(
            spec,
            reference,
            classified,
            pd.Series(scan_row),
            center_hr,
            output_suffix=f"{RUN_TAG}_rfbbp",
        )

    paths["metadata"].write_text(
        json.dumps(
            {
                "biomarker": spec.label,
                "data_file": str(spec.data_file),
                "train_fraction": args.train_frac,
                "validation_fraction": 1.0 - args.train_frac,
                "seed": args.seed,
                "run_tag": RUN_TAG,
                "reference_strategy": args.reference_strategy,
                "draws": args.draws,
                "tune": args.tune,
                "chains": args.chains,
                "target_accept": args.target_accept,
                "train_n": int(len(train)),
                "validation_n": int(len(validation)),
                "train_centres": int(train["center_id"].nunique()),
                "validation_centres": int(validation["center_id"].nunique()),
                "sofa_covariate": spec.sofa_col,
                "threshold_references": list(spec.references),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return pd.DataFrame(scan_rows), pd.concat(center_frames, ignore_index=True), split_out


def main() -> None:
    global OUTPUT_DIR, MODEL_DIR, TABLE_DIR, FIGURE_DIR, OUT_PREFIX, OUT_SCAN, OUT_CENTER_HR, OUT_SPLIT, RUN_TAG
    args = parse_args()
    train_percent = int(round(100 * args.train_frac))
    validation_percent = 100 - train_percent
    RUN_TAG = args.run_tag or f"train{train_percent}_validate{validation_percent}"
    OUTPUT_DIR = args.output_dir.resolve()
    MODEL_DIR = OUTPUT_DIR / "train_validate_models"
    TABLE_DIR = OUTPUT_DIR
    FIGURE_DIR = OUTPUT_DIR / "figures"
    OUT_PREFIX = f"three_biomarkers_{RUN_TAG}_rfbbp_threshold_transportability"
    OUT_SCAN = TABLE_DIR / f"{OUT_PREFIX}_scan.csv"
    OUT_CENTER_HR = TABLE_DIR / f"{OUT_PREFIX}_center_threshold_hr.csv"
    OUT_SPLIT = TABLE_DIR / f"{OUT_PREFIX}_split_assignment.csv"
    configure_style()
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
    selected_specs = [
        replace(spec, data_file=data_dir / spec.data_file.name)
        for spec in ALL_SPECS
        if args.biomarker in ("all", spec.key)
    ]
    if args.reference_set == "final":
        final_references = {"hemoglobin": (8.0, 12.0), "lactate": (4.0, 8.0), "creatinine": (1.5, 2.5)}
        selected_specs = [replace(spec, references=final_references[spec.key]) for spec in selected_specs]
    if args.references is not None:
        if args.biomarker == "all":
            raise ValueError("--references requires selecting one biomarker with --biomarker.")
        if any(value <= 0 for value in args.references):
            raise ValueError("All clinical reference thresholds must be positive.")
        selected_specs = [replace(spec, references=tuple(args.references)) for spec in selected_specs]
    if args.split_only:
        split_frames = []
        summary_rows = []
        for spec in selected_specs:
            split = split_within_center(load_data(spec), args.train_frac, args.seed)
            split_out = split[["stay_id", "database", "center_id", "time", "event", "exposure_analysis", "split"]].copy()
            split_out.insert(0, "biomarker", spec.label)
            split_frames.append(split_out)
            summary_rows.append({
                "biomarker": spec.label, "seed": args.seed, "train_fraction": args.train_frac,
                "total_n": len(split), "train_n": int(split.split.eq("train").sum()),
                "validation_n": int(split.split.eq("validation").sum()),
                "total_centres": split.center_id.nunique(),
                "train_centres": split.loc[split.split.eq("train"), "center_id"].nunique(),
                "validation_centres": split.loc[split.split.eq("validation"), "center_id"].nunique(),
            })
        pd.concat(split_frames, ignore_index=True).to_csv(OUT_SPLIT, index=False)
        pd.DataFrame(summary_rows).to_csv(TABLE_DIR / f"{OUT_PREFIX}_split_summary.csv", index=False)
        print(pd.DataFrame(summary_rows).to_string(index=False))
        print(f"Saved split assignment: {OUT_SPLIT}")
        return
    scan_frames = []
    center_frames = []
    split_frames = []
    for spec in selected_specs:
        scan, center_hr, split = run_one_spec(spec, args)
        scan_frames.append(scan)
        center_frames.append(center_hr)
        split_frames.append(split)

    scan_all = pd.concat(scan_frames, ignore_index=True)
    center_all = pd.concat(center_frames, ignore_index=True)
    split_all = pd.concat(split_frames, ignore_index=True)
    scan_all.to_csv(OUT_SCAN, index=False)
    center_all.to_csv(OUT_CENTER_HR, index=False)
    split_all.to_csv(OUT_SPLIT, index=False)

    print(f"Saved scan: {OUT_SCAN}")
    print(f"Saved center threshold HR table: {OUT_CENTER_HR}")
    print(f"Saved split assignment: {OUT_SPLIT}")
    print(
        scan_all[
            [
                "biomarker",
                "reference_value",
                "equivalent_threshold_min",
                "equivalent_threshold_max",
                "reclassified_percent",
                "common_centres_fitted",
                "raw_tau2",
                "calibrated_tau2",
                "tau2_absolute_change",
                "raw_I2_percent",
                "calibrated_I2_percent",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
