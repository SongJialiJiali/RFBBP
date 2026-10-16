from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np


@dataclass
class RFBBPPreparedData:
    S_raw: np.ndarray
    studies: np.ndarray
    study_idx: np.ndarray
    W: np.ndarray
    W_original: np.ndarray
    YT: np.ndarray
    Y1: np.ndarray
    Z: np.ndarray
    subject_long: np.ndarray
    study_long: np.ndarray
    interval_long: np.ndarray
    exposure_long: np.ndarray
    event_long: np.ndarray
    n: int
    n_studies: int
    n_intervals: int
    n_z: int
    standardization: dict[str, Any]


def _as_1d(x: Any, name: str, dtype: Any | None = None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape={arr.shape}.")
    return arr


def _prepare_z(Z: Any | None, n: int) -> np.ndarray:
    if Z is None:
        return np.zeros((n, 0), dtype=float)
    arr = np.asarray(Z, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Z must be two-dimensional, got shape={arr.shape}.")
    if arr.shape[0] != n:
        raise ValueError(f"Z has {arr.shape[0]} rows, expected {n}.")
    return arr


def _standardize_by_study(
    W: np.ndarray,
    study_idx: np.ndarray,
    n_studies: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    out = W.astype(float).copy()
    means = np.zeros(n_studies, dtype=float)
    sds = np.ones(n_studies, dtype=float)
    for s in range(n_studies):
        mask = study_idx == s
        values = W[mask]
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 1.0
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        out[mask] = (values - mean) / sd
        means[s] = mean
        sds[s] = sd
    return out, {"W_study_mean": means, "W_study_sd": sds}


def standardize_w_values(
    W: np.ndarray,
    study_idx: np.ndarray,
    n_studies: int,
    standardize_w: Literal["none", "global", "study", "study_global"],
) -> tuple[np.ndarray, dict[str, Any]]:
    if standardize_w == "none":
        return W.astype(float).copy(), {"standardize_w": "none"}

    if standardize_w == "global":
        mean = float(np.mean(W))
        sd = float(np.std(W, ddof=1)) if W.size > 1 else 1.0
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        return (W - mean) / sd, {"standardize_w": "global", "W_mean": mean, "W_sd": sd}

    if standardize_w == "study":
        out, info = _standardize_by_study(W, study_idx, n_studies)
        info["standardize_w"] = "study"
        return out, info

    if standardize_w == "study_global":
        study_scaled, info = _standardize_by_study(W, study_idx, n_studies)
        mean = float(np.mean(study_scaled))
        sd = float(np.std(study_scaled, ddof=1)) if study_scaled.size > 1 else 1.0
        if not np.isfinite(sd) or sd <= 0:
            sd = 1.0
        out = (study_scaled - mean) / sd
        info.update({"standardize_w": "study_global", "W_mean": mean, "W_sd": sd})
        return out, info

    raise ValueError(f"Unknown standardize_w={standardize_w!r}.")


def prepare_piecewise_survival_data(
    YT: Sequence[float],
    Y1: Sequence[int],
    cutpoints: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yt = _as_1d(YT, "YT", float)
    y1 = _as_1d(Y1, "Y1", int)
    if yt.size != y1.size:
        raise ValueError("YT and Y1 must have the same length.")
    if yt.size == 0:
        raise ValueError("Input data must contain at least one row.")
    if np.any(~np.isfinite(yt)) or np.any(yt <= 0):
        raise ValueError("YT must contain positive finite survival times.")
    if not np.isin(y1, [0, 1]).all():
        raise ValueError("Y1 must contain only 0/1 event indicators.")

    if cutpoints is None:
        bounds = np.asarray([0.0, float(np.max(yt))], dtype=float)
    else:
        bounds = np.asarray(cutpoints, dtype=float)
        if bounds.ndim != 1 or bounds.size < 2:
            raise ValueError("cutpoints must contain at least two ordered values.")
        if bounds[0] > 0:
            bounds = np.r_[0.0, bounds]
    if np.any(~np.isfinite(bounds)) or np.any(np.diff(bounds) <= 0):
        raise ValueError("cutpoints must be finite and strictly increasing.")

    horizon = float(bounds[-1])
    subject_rows: list[int] = []
    interval_rows: list[int] = []
    exposure_rows: list[float] = []
    event_rows: list[int] = []
    for i, (time, event) in enumerate(zip(yt, y1)):
        observed_until = min(float(time), horizon)
        for k in range(bounds.size - 1):
            start = float(bounds[k])
            stop = float(bounds[k + 1])
            dt = max(0.0, min(observed_until, stop) - start)
            if dt <= 0:
                continue
            subject_rows.append(i)
            interval_rows.append(k)
            exposure_rows.append(dt)
            event_rows.append(int(event == 1 and time <= horizon and start < time <= stop))

    return (
        np.asarray(subject_rows, dtype=int),
        np.asarray(interval_rows, dtype=int),
        np.asarray(exposure_rows, dtype=float),
        np.asarray(event_rows, dtype=int),
        bounds,
    )


def prepare_rfbbp_data(
    S: Sequence[Any],
    W: Sequence[float],
    YT: Sequence[float],
    Y1: Sequence[int],
    Z: Any | None = None,
    cutpoints: Sequence[float] | None = None,
    standardize_w: Literal["none", "global", "study", "study_global"] = "none",
) -> RFBBPPreparedData:
    s_raw = _as_1d(S, "S")
    w_original = _as_1d(W, "W", float)
    yt = _as_1d(YT, "YT", float)
    y1 = _as_1d(Y1, "Y1", int)
    n = w_original.size
    if not (s_raw.size == yt.size == y1.size == n):
        raise ValueError("S, W, YT, and Y1 must have the same length.")
    z = _prepare_z(Z, n)
    if np.any(~np.isfinite(w_original)):
        raise ValueError("W must contain finite values.")
    if np.any(~np.isfinite(z)):
        raise ValueError("Z must contain finite values.")

    studies, study_idx = np.unique(s_raw.astype(str), return_inverse=True)
    w, standardization = standardize_w_values(
        w_original,
        study_idx,
        int(studies.size),
        standardize_w,
    )
    subject_long, interval_long, exposure_long, event_long, _ = prepare_piecewise_survival_data(
        YT=yt,
        Y1=y1,
        cutpoints=cutpoints,
    )
    study_long = study_idx[subject_long]

    return RFBBPPreparedData(
        S_raw=s_raw,
        studies=studies,
        study_idx=study_idx.astype(int),
        W=w,
        W_original=w_original,
        YT=yt,
        Y1=y1,
        Z=z,
        subject_long=subject_long,
        study_long=study_long.astype(int),
        interval_long=interval_long,
        exposure_long=exposure_long,
        event_long=event_long,
        n=int(n),
        n_studies=int(studies.size),
        n_intervals=int(np.max(interval_long) + 1) if interval_long.size else 0,
        n_z=int(z.shape[1]),
        standardization=standardization,
    )
