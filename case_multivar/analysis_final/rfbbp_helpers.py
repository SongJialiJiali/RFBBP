from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg


CUTPOINTS = [1, 3, 7, 14, 28]
LOG_AUTO_VARIABLES = {"Lactate_hour_24", "Creatinine_hour_24", "Platelet_hour_24", "RAR_hour_24"}
BASIC_COVARIATES = ["age_z", "gender_male"]
SOFA_MAP = {
    "Lactate_hour_24": ("SOFA_score_hour_24", []),
    "Heart_rate_hour_24": ("SOFA_score_hour_24", []),
    "Temperature_hour_24": ("SOFA_score_hour_24", []),
    "Hemoglobin_hour_24": ("SOFA_score_hour_24", []),
    "RAR_hour_24": ("SOFA_score_hour_24", []),
    "Creatinine_hour_24": ("SOFA_no_renal", ["SOFA_score_hour_24", "Renal_score_hour_24"]),
    "Platelet_hour_24": ("SOFA_no_coagulation", ["SOFA_score_hour_24", "Coagulation_score_hour_24"]),
    "MAP_hour_24": ("SOFA_no_cardiovascular", ["SOFA_score_hour_24", "Cardiovascular_score_hour_24"]),
}

NEW_SCHEMA_ALIASES = {
    "dataset": "database",
    "mortality_28d_inhospital": "mortality_28d",
    "survival_time_28d_days": "survival_time",
    "height": "Height",
    "weight": "Weight",
    "bmi": "BMI",
    "albumin_first_0_24h": "Albumin_hour_24",
    "alt_first_0_24h": "ALT_hour_24",
    "ast_first_0_24h": "AST_hour_24",
    "bands_first_0_24h": "Bands_hour_24",
    "bicarbonate_first_0_24h": "Bicarbonate_hour_24",
    "bilirubin_first_0_24h": "Bilirubin_hour_24",
    "bun_first_0_24h": "BUN_hour_24",
    "chloride_first_0_24h": "Chloride_hour_24",
    "creatinine_first_0_24h": "Creatinine_hour_24",
    "crp_first_0_24h": "CRP_hour_24",
    "fio2_first_0_24h": "FiO2_hour_24",
    "gcs_first_0_24h": "GCS_hour_24",
    "glucose_first_0_24h": "Glucose_hour_24",
    "heart_rate_first_0_24h": "Heart_rate_hour_24",
    "hemoglobin_first_0_24h": "Hemoglobin_hour_24",
    "inr_first_0_24h": "INR_hour_24",
    "lactate_first_0_24h": "Lactate_hour_24",
    "lymphocyte_percent_first_0_24h": "Lymphocyte_percent_hour_24",
    "map_first_0_24h": "MAP_hour_24",
    "pao2_first_0_24h": "PaO2_hour_24",
    "platelet_first_0_24h": "Platelet_hour_24",
    "rdw_first_0_24h": "RDW_hour_24",
    "respiratory_rate_first_0_24h": "Respiratory_rate_hour_24",
    "so2_first_0_24h": "SO2_hour_24",
    "sodium_first_0_24h": "Sodium_hour_24",
    "systolic_abp_first_0_24h": "Systolic_ABP_hour_24",
    "temperature_first_0_24h": "Temperature_hour_24",
    "troponin_t_first_0_24h": "Troponin T_hour_24",
    "urine_sum_0_24h": "Urine_hour_24",
    "dobutamine_first_positive_0_24h": "Dobutamine_hour_24",
    "dopamine_first_positive_0_24h": "Dopamine_hour_24",
    "epinephrine_first_positive_0_24h": "Epinephrine_hour_24",
    "norepinephrine_first_positive_0_24h": "Norepinephrine_hour_24",
    "phenylephrine_first_positive_0_24h": "Phenylephrine_hour_24",
    "vasopressin_first_positive_0_24h": "Vasopressin_hour_24",
    "ventilation_any_0_24h": "Ventilation_hour_24",
    "sofa_respiration_0_24h": "Respiration_score_hour_24",
    "sofa_coagulation_0_24h": "Coagulation_score_hour_24",
    "sofa_liver_0_24h": "Liver_score_hour_24",
    "sofa_cardiovascular_0_24h": "Cardiovascular_score_hour_24",
    "sofa_cns_0_24h": "CNS_score_hour_24",
    "sofa_renal_0_24h": "Renal_score_hour_24",
    "sofa_total_0_24h": "SOFA_score_hour_24",
}


def set_runtime_env() -> None:
    for name in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ]:
        os.environ.setdefault(name, "1")
    os.environ.setdefault("ARVIZ_DATA", "/tmp/arviz")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_rfbbp_simple")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_rfbbp_simple")
    os.environ.setdefault("PYTENSOR_FLAGS", "base_compiledir=/tmp/pytensor_rfbbp_simple,compile__timeout=600,compile__wait=5")
    os.environ.setdefault("AESARA_FLAGS", "base_compiledir=/tmp/aesara_rfbbp_simple")
    Path(os.environ["ARVIZ_DATA"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path("/tmp/pytensor_rfbbp_simple").mkdir(parents=True, exist_ok=True)
    Path("/tmp/aesara_rfbbp_simple").mkdir(parents=True, exist_ok=True)


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def zscore(series: pd.Series) -> pd.Series:
    x = numeric(series)
    sd = x.std(skipna=True, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(0.0, index=x.index)
    return (x - x.mean(skipna=True)) / sd


def encode_gender(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    raw = series.astype("string").str.strip().str.lower()
    mapping = {
        "m": 1,
        "male": 1,
        "1": 1,
        "1.0": 1,
        "f": 0,
        "female": 0,
        "0": 0,
        "0.0": 0,
    }
    encoded = raw.map(mapping).astype("float")
    info = {
        "encoded_column": "gender_male",
        "mapping": {"Male/M/1": 1, "Female/F/0": 0},
        "raw_values": sorted(series.dropna().astype(str).unique().tolist()),
        "unmapped_values": sorted(series[encoded.isna() & series.notna()].astype(str).unique().tolist()),
    }
    return encoded, info


def standardize_input_schema(df: pd.DataFrame, source_label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    alias_applied: dict[str, str] = {}
    for source_col, target_col in NEW_SCHEMA_ALIASES.items():
        if target_col not in out.columns and source_col in out.columns:
            out[target_col] = out[source_col]
            alias_applied[source_col] = target_col

    if "database" not in out.columns:
        out["database"] = source_label
    out["database"] = out["database"].astype("string").str.strip().str.lower()
    out["database"] = out["database"].replace({"mimic": "mimic_iv", "mimic-iv": "mimic_iv"})
    out.loc[out["database"].isna(), "database"] = source_label

    if "center_id" not in out.columns:
        if source_label == "mimic_iv":
            out["center_id"] = "mimic_iv"
        elif "hospital_id" in out.columns:
            hospital = numeric(out["hospital_id"]).astype("Int64").astype(str)
            out["center_id"] = "eicu_" + hospital
            out.loc[hospital.eq("<NA>"), "center_id"] = np.nan
        else:
            out["center_id"] = np.nan
    out["center_id"] = out["center_id"].astype("string").str.strip()
    out["center_id"] = out["center_id"].str.replace(":", "_", regex=False)
    out.loc[out["center_id"].eq("mimic"), "center_id"] = "mimic_iv"
    out.loc[out["center_id"].isin(["", "<NA>", "nan", "None"]), "center_id"] = np.nan

    if "is_baseline_vent" not in out.columns:
        out["is_baseline_vent"] = 0.0

    return out, {
        "source_label": source_label,
        "input_shape": df.shape,
        "standardized_shape": out.shape,
        "alias_applied": alias_applied,
    }


def _prepare_merged_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = raw.copy()
    required_cols = ["mortality_28d", "survival_time", "gender"]
    missing_required = [col for col in required_cols if col not in merged.columns]
    if missing_required:
        raise KeyError(f"Missing required columns after schema standardization: {missing_required}")

    merged["event"] = numeric(merged["mortality_28d"])
    merged["event"] = merged["event"].where(merged["event"].isin([0, 1]))
    merged["time"] = numeric(merged["survival_time"])

    if "RAR_hour_24" not in merged.columns and {"RDW_hour_24", "Albumin_hour_24"}.issubset(merged.columns):
        rdw = numeric(merged["RDW_hour_24"])
        albumin = numeric(merged["Albumin_hour_24"])
        merged["RAR_hour_24"] = np.where((rdw > 0) & (albumin > 0), rdw / albumin, np.nan)

    gender, gender_info = encode_gender(merged["gender"])
    merged["gender_male"] = gender

    protected = {"database", "center_id", "gender"}
    for col in merged.columns:
        if col not in protected:
            merged[col] = numeric(merged[col])
    merged = merged.replace([np.inf, -np.inf], np.nan)
    merged = merged[merged["event"].isin([0, 1]) & merged["time"].gt(0) & merged["center_id"].notna()].copy()
    merged["event"] = merged["event"].astype(int)
    info = {
        "merged_shape": merged.shape,
        "gender": gender_info,
    }
    return merged, info


def read_and_merge_data(mimic_path: Path, eicu_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if eicu_path is None:
        combined_raw = pd.read_csv(mimic_path, low_memory=False)
        combined, combined_info = standardize_input_schema(combined_raw, "combined")
        merged, info = _prepare_merged_data(combined)
        info.update(
            {
                "input_mode": "combined",
                "combined_shape": combined.shape,
                "common_columns_n": len(combined.columns),
                "schema": {
                    "combined": combined_info,
                    "outcome": "mortality_28d_inhospital -> mortality_28d/event when using the new schema",
                    "time": "survival_time_28d_days -> survival_time/time when using the new schema",
                },
            }
        )
        return merged, info

    mimic_raw = pd.read_csv(mimic_path, low_memory=False)
    eicu_raw = pd.read_csv(eicu_path, low_memory=False)
    mimic, mimic_info = standardize_input_schema(mimic_raw, "mimic_iv")
    eicu, eicu_info = standardize_input_schema(eicu_raw, "eicu")

    common_cols = [c for c in mimic.columns if c in eicu.columns]
    combined = pd.concat([mimic[common_cols], eicu[common_cols]], ignore_index=True, sort=False)
    merged, info = _prepare_merged_data(combined)
    info.update(
        {
            "input_mode": "separate_sources",
            "mimic_shape": mimic.shape,
            "eicu_shape": eicu.shape,
            "common_columns_n": len(common_cols),
            "schema": {
                "mimic": mimic_info,
                "eicu": eicu_info,
                "outcome": "mortality_28d_inhospital -> mortality_28d/event when using the new schema",
                "time": "survival_time_28d_days -> survival_time/time when using the new schema",
            },
        }
    )
    return merged, info


def add_sofa_adjusted_variables(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required = {
        "SOFA_no_renal": ("SOFA_score_hour_24", "Renal_score_hour_24"),
        "SOFA_no_coagulation": ("SOFA_score_hour_24", "Coagulation_score_hour_24"),
        "SOFA_no_cardiovascular": ("SOFA_score_hour_24", "Cardiovascular_score_hour_24"),
    }
    for new_col, (total, component) in required.items():
        if total in out.columns and component in out.columns:
            out[new_col] = numeric(out[total]) - numeric(out[component])
    return out


def covariates_for(variable: str, adjustment: str) -> list[str]:
    if adjustment == "basic":
        return BASIC_COVARIATES.copy()
    if adjustment != "sofa_adjusted":
        raise ValueError("adjustment must be 'basic' or 'sofa_adjusted'")
    sofa_col, needed = SOFA_MAP[variable]
    missing = [c for c in needed if c not in SOFA_MAP and False]
    if missing:
        raise KeyError(f"Missing SOFA component columns for {variable}: {missing}")
    return BASIC_COVARIATES + [f"{sofa_col}_z"]


def build_variable_dataset(
    df: pd.DataFrame,
    variable: str,
    adjustment: str = "basic",
    transform: str = "auto",
    winsorize: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if variable not in df.columns:
        raise KeyError(f"{variable} is not present in the merged data")
    if adjustment not in {"basic", "sofa_adjusted"}:
        raise ValueError("adjustment must be 'basic' or 'sofa_adjusted'")

    sofa_col = None
    if adjustment == "sofa_adjusted":
        sofa_col, needed = SOFA_MAP[variable]
        missing = [c for c in needed if c not in df.columns]
        if sofa_col not in df.columns:
            missing.append(sofa_col)
        if missing:
            raise KeyError(f"Missing SOFA columns for {variable}: {sorted(set(missing))}")

    id_cols = [c for c in ["stay_id", "subject_id", "hadm_id"] if c in df.columns]
    cols = id_cols + ["database", "center_id", "time", "event", variable, "age", "gender_male"]
    if sofa_col:
        cols.append(sofa_col)
    out = df.loc[:, list(dict.fromkeys(cols))].copy()
    out = out.rename(columns={variable: "exposure_raw"})
    out["exposure_raw"] = numeric(out["exposure_raw"])
    exposure = out["exposure_raw"].copy()
    if winsorize and exposure.notna().sum() > 0:
        q01, q99 = exposure.quantile([0.01, 0.99])
        exposure = exposure.clip(q01, q99)

    nonpositive_to_missing = 0
    applied_transform = "raw"
    if transform == "auto" and variable in LOG_AUTO_VARIABLES:
        nonpositive_to_missing = int(exposure.notna().sum() - exposure.where(exposure > 0).notna().sum())
        exposure = np.log(exposure.where(exposure > 0))
        applied_transform = "log"
    elif transform == "raw" or transform == "auto":
        applied_transform = "raw"
    else:
        raise ValueError("transform must be 'auto' or 'raw'")

    out["exposure_analysis"] = exposure
    out["age_z"] = zscore(out["age"])
    covariates = BASIC_COVARIATES.copy()
    if sofa_col:
        zcol = f"{sofa_col}_z"
        out[zcol] = zscore(out[sofa_col])
        covariates.append(zcol)
    needed = ["database", "center_id", "time", "event", "exposure_raw", "exposure_analysis", *covariates]
    before = len(out)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=needed).copy()
    out = out[out["time"].gt(0) & out["event"].isin([0, 1])].copy()
    out["event"] = out["event"].astype(int)
    info = {
        "variable": variable,
        "adjustment": adjustment,
        "transform": applied_transform,
        "winsorize": winsorize,
        "nonpositive_to_missing_n": nonpositive_to_missing,
        "complete_case_dropped_n": int(before - len(out)),
        "covariates": covariates,
    }
    out.attrs.update(info)
    return out, info


def center_description(d: pd.DataFrame, variable: str, n_min: int, event_min: int, nonevent_min: int) -> pd.DataFrame:
    rows = []
    for (database, center_id), g in d.groupby(["database", "center_id"], dropna=False):
        x = numeric(g["exposure_raw"])
        q1, q3 = x.quantile([0.25, 0.75]) if x.notna().sum() else (np.nan, np.nan)
        rows.append(
            {
                "variable": variable,
                "database": database,
                "center_id": center_id,
                "n": int(len(g)),
                "event_n": int(g["event"].sum()),
                "nonevent_n": int(len(g) - g["event"].sum()),
                "event_rate": float(g["event"].mean()),
                "exposure_nonmissing_n": int(x.notna().sum()),
                "exposure_mean": float(x.mean(skipna=True)),
                "exposure_sd": float(x.std(skipna=True, ddof=1)),
                "exposure_median": float(x.median(skipna=True)),
                "exposure_iqr": float(q3 - q1) if np.isfinite(q1) and np.isfinite(q3) else np.nan,
                "exposure_min": float(x.min(skipna=True)),
                "exposure_max": float(x.max(skipna=True)),
            }
        )
    out = pd.DataFrame(rows)
    out["eligible"] = out["n"].ge(n_min) & out["event_n"].ge(event_min) & out["nonevent_n"].ge(nonevent_min)
    return out.sort_values(["eligible", "n"], ascending=[False, False]).reset_index(drop=True)


def eligible_dataset(d: pd.DataFrame, center_table: pd.DataFrame) -> pd.DataFrame:
    centers = center_table.loc[center_table["eligible"], "center_id"].astype(str)
    return d[d["center_id"].astype(str).isin(centers)].copy()


def _cox(d: pd.DataFrame, exposure_col: str, covariates: list[str], strata_center: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = ["time", "event", exposure_col, *covariates]
    work = d.dropna(subset=cols).copy()
    dropped_constant_covariates = [
        col for col in covariates if col in work.columns and numeric(work[col]).nunique(dropna=True) <= 1
    ]
    model_covariates = [col for col in covariates if col not in dropped_constant_covariates]
    exog_cols = [exposure_col, *model_covariates]
    if len(work) == 0 or work["event"].nunique() < 2:
        return pd.DataFrame([{"term": exposure_col, "status": "skipped_insufficient_data"}]), {
            "status": "skipped_insufficient_data",
            "n": int(len(work)),
            "events": int(work["event"].sum()) if len(work) else 0,
            "exog_columns": exog_cols,
            "dropped_constant_covariates": dropped_constant_covariates,
        }
    if numeric(work[exposure_col]).nunique(dropna=True) <= 1:
        return pd.DataFrame([{"term": exposure_col, "status": "failed", "error": "constant_exposure"}]), {
            "status": "failed",
            "error": "constant_exposure",
            "n": int(len(work)),
            "events": int(work["event"].sum()),
            "exog_columns": exog_cols,
            "dropped_constant_covariates": dropped_constant_covariates,
        }
    try:
        strata_values = work["center_id"].astype(str).to_numpy() if strata_center else None
        fit = PHReg(
            work["time"].to_numpy(float),
            work[exog_cols].to_numpy(float),
            status=work["event"].to_numpy(int),
            strata=strata_values,
            ties="efron",
        ).fit(disp=False, maxiter=100)
        ci = fit.conf_int()
        rows = []
        for i, term in enumerate(exog_cols):
            rows.append(
                {
                    "term": term,
                    "coef": float(fit.params[i]),
                    "se": float(fit.bse[i]),
                    "hr": float(np.exp(fit.params[i])),
                    "ci_low": float(np.exp(ci[i, 0])),
                    "ci_high": float(np.exp(ci[i, 1])),
                    "p": float(fit.pvalues[i]),
                    "status": "ok",
                }
            )
        return pd.DataFrame(rows), {
            "status": "ok",
            "n": int(len(work)),
            "events": int(work["event"].sum()),
            "exog_columns": exog_cols,
            "dropped_constant_covariates": dropped_constant_covariates,
        }
    except Exception as exc:
        return pd.DataFrame([{"term": exposure_col, "status": "failed", "error": repr(exc)}]), {
            "status": "failed",
            "error": repr(exc),
            "n": int(len(work)),
            "events": int(work["event"].sum()),
            "exog_columns": exog_cols,
            "dropped_constant_covariates": dropped_constant_covariates,
        }


def fit_pooled_cox(d: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _cox(d, "exposure_analysis", covariates, strata_center=False)


def fit_stratified_cox(d: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _cox(d, "exposure_analysis", covariates, strata_center=True)


def fit_global_z_cox(d: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = d.copy()
    sd = work["exposure_analysis"].std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.DataFrame([{"term": "exposure_global_z", "status": "failed", "error": "nonpositive_exposure_sd"}]), {
            "status": "failed",
            "error": "nonpositive_exposure_sd",
            "n": int(len(work)),
            "events": int(work["event"].sum()),
            "exog_columns": ["exposure_global_z", *covariates],
        }
    work["exposure_global_z"] = (work["exposure_analysis"] - work["exposure_analysis"].mean()) / sd
    return _cox(work, "exposure_global_z", covariates, strata_center=False)


def add_center_z_exposure(d: pd.DataFrame) -> pd.DataFrame:
    work = d.copy()
    work["exposure_center_z"] = work.groupby("center_id")["exposure_analysis"].transform(lambda s: (s - s.mean()) / s.std(ddof=1))
    return work


def fit_center_z_cox(d: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = add_center_z_exposure(d)
    return _cox(work, "exposure_center_z", covariates, strata_center=False)


def fit_within_center_z_cox(d: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = add_center_z_exposure(d)
    return _cox(work, "exposure_center_z", covariates, strata_center=True)


def fit_two_stage_meta(d: pd.DataFrame, covariates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    center_rows = []
    for center_id, g in d.groupby("center_id", dropna=False):
        work = g.copy()
        sd = work["exposure_analysis"].std(ddof=1)
        if not np.isfinite(sd) or sd <= 0:
            center_rows.append({"center_id": center_id, "status": "skipped_zero_sd", "n": len(work), "events": int(work["event"].sum())})
            continue
        work["exposure_center_z"] = (work["exposure_analysis"] - work["exposure_analysis"].mean()) / sd
        terms, meta = _cox(work, "exposure_center_z", covariates, strata_center=False)
        row = {"center_id": center_id, "n": meta["n"], "events": meta["events"], "status": meta["status"]}
        if "term" in terms.columns:
            exp_row = terms[terms["term"].eq("exposure_center_z")].head(1)
        else:
            exp_row = pd.DataFrame()
        if not exp_row.empty and meta["status"] == "ok":
            row.update(exp_row.iloc[0].to_dict())
        elif "error" in meta:
            row["error"] = meta["error"]
        center_rows.append(row)
    center_df = pd.DataFrame(center_rows)
    for col in ["coef", "se"]:
        if col not in center_df.columns:
            center_df[col] = np.nan
    ok = center_df[(center_df["status"].eq("ok")) & center_df["coef"].notna() & center_df["se"].gt(0)].copy()
    if len(ok) == 0:
        meta_df = pd.DataFrame([{"status": "failed_no_center_models"}])
        return center_df, meta_df
    yi = ok["coef"].to_numpy(float)
    vi = np.square(ok["se"].to_numpy(float))
    wi = 1 / vi
    fixed = float(np.sum(wi * yi) / np.sum(wi))
    fixed_se = math.sqrt(float(1 / np.sum(wi)))
    fixed_ci_low, fixed_ci_high = fixed - 1.96 * fixed_se, fixed + 1.96 * fixed_se
    q = float(np.sum(wi * (yi - fixed) ** 2))
    c = float(np.sum(wi) - np.sum(wi**2) / np.sum(wi))
    tau2 = max(0.0, (q - (len(yi) - 1)) / c) if c > 0 and len(yi) > 1 else 0.0
    w_re = 1 / (vi + tau2)
    pooled = float(np.sum(w_re * yi) / np.sum(w_re))
    se = math.sqrt(float(1 / np.sum(w_re)))
    ci_low, ci_high = pooled - 1.96 * se, pooled + 1.96 * se
    i2 = max(0.0, (q - (len(yi) - 1)) / q) * 100 if q > 0 and len(yi) > 1 else 0.0
    pred_se = math.sqrt(se**2 + tau2)
    pred_low, pred_high = pooled - 1.96 * pred_se, pooled + 1.96 * pred_se
    meta_df = pd.DataFrame(
        [
            {
                "status": "ok",
                "meta_model": "fixed_effect",
                "exposure_scale": "center_z",
                "center_models_n": int(len(ok)),
                "coef": fixed,
                "se": fixed_se,
                "hr": float(np.exp(fixed)),
                "ci_low": float(np.exp(fixed_ci_low)),
                "ci_high": float(np.exp(fixed_ci_high)),
                "tau2": 0.0,
                "I2": i2,
                "prediction_low": np.nan,
                "prediction_high": np.nan,
                "Q": q,
                "Q_p": float(1 - stats.chi2.cdf(q, len(yi) - 1)) if len(yi) > 1 else np.nan,
            },
            {
                "status": "ok",
                "meta_model": "random_effect",
                "exposure_scale": "center_z",
                "center_models_n": int(len(ok)),
                "coef": pooled,
                "se": se,
                "hr": float(np.exp(pooled)),
                "ci_low": float(np.exp(ci_low)),
                "ci_high": float(np.exp(ci_high)),
                "tau2": tau2,
                "I2": i2,
                "prediction_low": float(np.exp(pred_low)),
                "prediction_high": float(np.exp(pred_high)),
                "Q": q,
                "Q_p": float(1 - stats.chi2.cdf(q, len(yi) - 1)) if len(yi) > 1 else np.nan,
            }
        ]
    )
    return center_df, meta_df


def fit_rfbbp(
    d: pd.DataFrame,
    variable: str,
    adjustment: str,
    covariates: list[str],
    output_dir: Path,
    random_seed: int = 3000,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.95,
) -> dict[str, Any]:
    set_runtime_env()
    project_root = Path(__file__).resolve().parents[2]
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from rfbbp.rfbbp import RFBBPPyMC

    work = d.dropna(subset=["database", "center_id", "time", "event", "exposure_analysis", *covariates]).reset_index(drop=True)
    if work["center_id"].nunique() < 2:
        raise ValueError("RFBBP requires at least 2 eligible centers.")

    s = work["center_id"].astype(str).to_numpy()
    w = work["exposure_analysis"].to_numpy(dtype=float)
    yt = work["time"].to_numpy(dtype=float)
    y1 = work["event"].to_numpy(dtype=int)
    z = work[covariates].to_numpy(dtype=float)
    cutpoints = [0.0, *[float(x) for x in CUTPOINTS]]

    model_kwargs = {
        "cutpoints": cutpoints,
        "auto_cutpoints": False,
        "study_event_cutpoint_constraint": False,
        "max_cuts": 15,
        "min_events_per_interval": 10,
        "hierarchical_calibration": True,
        "betax_sigma0": 2.5,
        "mu_a_sigma0": 0.5,
        "tau_a_sigma0": 0.5,
        "mu_l_sigma0": 0.5,
        "tau_l_sigma0": 0.5,
        "sigma_w_prior": "invgamma",
        "sigma_w2_alpha0": 3.0,
        "sigma_w2_gamma0": 1.0,
        "standardize_w": "none",
        "solver": "pymc",
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "target_accept": target_accept,
        "seed": random_seed,
        "pbar": True,
    }

    model = RFBBPPyMC(**model_kwargs)
    model.fit(S=s, W=w, YT=yt, Y1=y1, Z=z)

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
    summary = model.summary(var_names=summary_vars).reset_index().rename(columns={"index": "parameter"})
    diagnostics_series = model.diagnostics()

    tables_dir = output_dir / "tables"
    models_dir = output_dir / "models"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{variable}_{adjustment}_rfbbp_hierarchical_5interval"
    summary_path = tables_dir / f"{stem}_posterior_summary.csv"
    diagnostics_path = tables_dir / f"{stem}_diagnostics.csv"
    metadata_path = tables_dir / f"{stem}_metadata.json"
    nc_path = models_dir / f"{stem}.nc"

    summary.to_csv(summary_path, index=False)
    diagnostics_series.to_csv(diagnostics_path)
    model.idata_.to_netcdf(nc_path)

    hr_row = summary.loc[summary["parameter"].eq("HR_x")].iloc[0]
    beta_row = summary.loc[summary["parameter"].eq("beta_x")].iloc[0]
    diagnostics = {
        "variable": variable,
        "adjustment": adjustment,
        "analysis": "formal",
        "baseline_prior": "noncentered_random_walk",
        "standardize_w": "none",
        "n": int(len(work)),
        "events": int(work["event"].sum()),
        "centers": int(work["center_id"].nunique()),
        "center_ids": sorted(work["center_id"].astype(str).unique().tolist()),
        "cutpoints": cutpoints,
        "covariates": covariates,
        "divergent_transitions": float(diagnostics_series.get("diag_n_divergent", np.nan)),
        "max_rhat": float(diagnostics_series.get("diag_max_r_hat", np.nan)),
        "min_ess_bulk": float(diagnostics_series.get("diag_min_ess_bulk", np.nan)),
        "min_ess_tail": float(diagnostics_series.get("diag_min_ess_tail", np.nan)),
        "runtime_seconds": float(diagnostics_series.get("runtime_seconds", np.nan)),
        "summary_csv": str(summary_path),
        "diagnostics_csv": str(diagnostics_path),
        "metadata_json": str(metadata_path),
        "netcdf": str(nc_path),
    }
    result = {
        **diagnostics,
        "beta_x_mean": float(beta_row["estimate"]),
        "beta_x_cri_low": float(beta_row["CI_1"]),
        "beta_x_cri_high": float(beta_row["CI_2"]),
        "HR_x_mean": float(hr_row["estimate"]),
        "HR_x_cri_low": float(hr_row["CI_1"]),
        "HR_x_cri_high": float(hr_row["CI_2"]),
    }
    metadata_path.write_text(
        json.dumps(
            {
                "variable": variable,
                "adjustment": adjustment,
                "data_file": f"analysis_datasets/{variable}_{adjustment}.csv",
                "n_rows": int(len(work)),
                "n_centers": int(work["center_id"].nunique()),
                "n_events": int(y1.sum()),
                "covariates": covariates,
                "model_kwargs": model_kwargs,
                "selected_cutpoints": [float(x) for x in np.asarray(model.cutpoints_).ravel()],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "model": model,
        "idata": model.idata_,
        "summary": summary,
        "result": result,
        "diagnostics": diagnostics,
        "netcdf": nc_path,
    }
