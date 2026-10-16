import os
import os.path as osp
import sys
import multiprocessing as mp
import re
import json
import glob
from typing import Literal, Sequence
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import asdict
import tempfile

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg

import sys
from rfbbp.simulate import (
    ExpSimulator,
    GompertzSimulator,
    LogLogisticSimulator,
    PEMSimulator,
    WeibSimulator,
    default_serializer,
)
from rfbbp.functions import Timer

import ipdb

def init_worker():
    import os
    import tempfile

    pid = os.getpid()

    pytensor_compiledir = os.path.join(
        tempfile.gettempdir(),
        f"pytensor_compiledir_{pid}"
    )
    os.makedirs(pytensor_compiledir, exist_ok=True)

    xdg_cache_home = os.path.join(
        tempfile.gettempdir(),
        f"xdg_cache_{pid}"
    )
    os.makedirs(xdg_cache_home, exist_ok=True)

    os.environ["XDG_CACHE_HOME"] = xdg_cache_home

    os.environ["PYTENSOR_FLAGS"] = (
        f"compiledir={pytensor_compiledir},"
        f"compile__timeout=600,"
        f"compile__wait=5"
    )


def resolve_simulation_data_dir(data_dir: str, analyzed_dir: str) -> str:
    """Resolve historical relative data_dir values saved in analysis params."""
    candidates = []
    if osp.isabs(data_dir):
        candidates.append(data_dir)
    else:
        candidates.append(data_dir)
        analyzed_abs = osp.abspath(analyzed_dir)
        parent = analyzed_abs
        while True:
            candidates.append(osp.join(parent, data_dir))
            next_parent = osp.dirname(parent)
            if next_parent == parent:
                break
            parent = next_parent

    for candidate in candidates:
        if osp.exists(osp.join(candidate, "params.json")):
            return candidate

    raise FileNotFoundError(
        "Unable to resolve simulation data_dir from analysis params. "
        f"data_dir={data_dir!r}, analyzed_dir={analyzed_dir!r}. "
        "Tried:\n" + "\n".join(candidates)
    )
    
def method_xtrue_c(
    X: np.ndarray,
    YT: np.ndarray,
    Y1: np.ndarray,
    Z: np.ndarray | None,
) -> np.ndarray:

    X  = X.reshape(-1)
    YT = YT.reshape(-1).astype(float)
    Y1 = Y1.reshape(-1).astype(int)
    if Z is not None:
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        X = np.column_stack([X, Z])
    model = PHReg(endog=YT, exog=X, status=Y1, ties="efron")
    res = model.fit()
    return np.r_[res.params[0], res.conf_int()[0, :]]

def method_xtrue_cs(
    X: np.ndarray,
    YT: np.ndarray,
    Y1: np.ndarray,
    Z: np.ndarray | None,
    S: np.ndarray,
) -> np.ndarray:

    X  = X.reshape(-1)
    YT = YT.reshape(-1).astype(float)
    Y1 = Y1.reshape(-1).astype(int)
    S = np.asarray(S).reshape(-1)
    if Z is not None:
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        X = np.column_stack([X, Z])
    model = PHReg(endog=YT, exog=X, status=Y1, strata=S, ties="efron")
    res = model.fit()
    return np.r_[res.params[0], res.conf_int()[0, :]]

def method_xtrue_ps(
    X: np.ndarray,
    YT: np.ndarray,
    Y1: np.ndarray,
    Z: np.ndarray | None,
    S: np.ndarray,
) -> np.ndarray:

    X  = X.reshape(-1)
    YT = YT.reshape(-1).astype(float)
    Y1 = Y1.reshape(-1).astype(int)
    S = np.asarray(S).reshape(-1)
    if Z is not None:
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        X = np.column_stack([X, Z])
    
    S_unique = np.sort(np.unique(S))
    S_dummy = np.column_stack([(S == s).astype(float) for s in S_unique])
    exog = np.column_stack([X, S_dummy])
    offset = np.log(np.maximum(YT, 1e-12))
    model = sm.GLM(Y1, exog, family=sm.families.Poisson(), offset=offset)
    res = model.fit()
    return np.r_[res.params[0], res.conf_int()[0, :]]

def method_naive(W, YT, Y1, Z, ) -> np.ndarray:
    W  = W.reshape(-1)
    YT = YT.reshape(-1).astype(float)
    Y1 = Y1.reshape(-1).astype(int)
    if Z is not None:
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        W = np.column_stack([W, Z])
    model = PHReg(endog=YT, exog=W, status=Y1, ties="efron")
    res = model.fit()
    return np.r_[res.params[0], res.conf_int()[0, :]]

def method_naive_s(W, YT, Y1, Z, S) -> np.ndarray:
    W  = W.reshape(-1)
    YT = YT.reshape(-1).astype(float)
    Y1 = Y1.reshape(-1).astype(int)
    S = np.asarray(S).reshape(-1)
    if Z is not None:
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        W = np.column_stack([W, Z])
    model = PHReg(endog=YT, exog=W, status=Y1, strata=S, ties="efron")
    res = model.fit()
    return np.r_[res.params[0], res.conf_int()[0, :]]

def method_wstandar(W, YT, Y1, Z, S) -> np.ndarray:
    W  = W.reshape(-1).astype(float)
    YT = YT.reshape(-1).astype(float)
    Y1 = Y1.reshape(-1).astype(int)
    S = np.asarray(S).reshape(-1)
    
    W_std = W.copy()
    for s in np.unique(S):
        Ws = W[S == s]
        mean_s = Ws.mean()
        sd_s = Ws.std(ddof=1)
        W_std[S == s] = (Ws - mean_s) / sd_s
    
    if Z is not None:
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        W_std = np.column_stack([W_std, Z])
        
    model = PHReg(endog=YT, exog=W_std, status=Y1, ties="efron")
    res = model.fit()
    return np.r_[res.params[0], res.conf_int()[0, :]]

def method_wstandar_s(W, YT, Y1, Z, S) -> np.ndarray:
    W  = W.reshape(-1).astype(float)
    YT = YT.reshape(-1).astype(float)
    Y1 = Y1.reshape(-1).astype(int)
    S = np.asarray(S).reshape(-1)
    
    W_std = W.copy()
    for s in np.unique(S):
        Ws = W[S == s]
        mean_s = Ws.mean()
        sd_s = Ws.std(ddof=1)
        W_std[S == s] = (Ws - mean_s) / sd_s
    
    if Z is not None:
        Z = Z.reshape(-1, 1) if Z.ndim == 1 else Z
        W_std = np.column_stack([W_std, Z])
        
    model = PHReg(endog=YT, exog=W_std, status=Y1, strata=S, ties="efron")
    res = model.fit()
    return np.r_[res.params[0], res.conf_int()[0, :]]



def analyze_data(
    X: np.ndarray,
    YT: np.ndarray,
    Y1: np.ndarray,
    W: np.ndarray,
    S: np.ndarray | None,
    Z: np.ndarray | None,
    ncores: int,
    methods: Sequence[Literal["xtrue_c", "xtrue_cs", "xtrue_ps", "wstandar", "wstandar_s", "naive", "naive_s", "rfbbp", "rfbbp_mcmc", "rfbbp_cox", "rfbbpcox", "rfbbp_vii", "rfbbp_vbn"]],
    rfbbp_class_kwargs: dict,
    rfbbp_fit_kwargs: dict,
    vi_class_kwargs: dict | None = None,
    vbn_class_kwargs: dict | None = None,
    export_x_mean: bool = False,
) -> tuple[dict[str, np.ndarray], pd.DataFrame | None]:
    
    
    res = {}
    x_mean_df = None
    result_columns = ["estimate", "post_sd", "CI_1", "CI_2"]
    for methodi in methods:
        diag = None
        with Timer() as t:
            try:
                if methodi == "xtrue_c":
                    resi = method_xtrue_c(X, YT, Y1, Z)
                elif methodi == "xtrue_cs":
                    resi = method_xtrue_cs(X, YT, Y1, Z, S)
                elif methodi == "xtrue_ps":
                    resi = method_xtrue_ps(X, YT, Y1, Z, S)
                elif methodi == "wstandar":
                    resi = method_wstandar(W, YT, Y1, Z, S)
                elif methodi == "wstandar_s":
                    resi = method_wstandar_s(W, YT, Y1, Z, S)
                elif methodi == "naive":
                    resi = method_naive(W, YT, Y1, Z)
                elif methodi == "naive_s":
                    resi = method_naive_s(W, YT, Y1, Z, S)
                elif methodi in {"rfbbp", "rfbbp_mcmc"}:
                    from rfbbp import RFBBPPyMC
                    estimator = RFBBPPyMC(**rfbbp_class_kwargs)
                    estimator.fit(S, W, YT, Y1, Z, **rfbbp_fit_kwargs)
                    resi = estimator.summary()
                    diag = estimator.diagnostics()
                    if export_x_mean:
                        x_mean = estimator.posterior_x_mean()
                        x_mean_df = pd.DataFrame(
                            {
                                "subject_id": np.arange(x_mean.shape[0]),
                                "S": np.asarray(S).reshape(-1),
                                "W": np.asarray(W).reshape(-1),
                                "YT": np.asarray(YT).reshape(-1),
                                "Y1": np.asarray(Y1).reshape(-1),
                                "X_true": np.asarray(X).reshape(-1),
                                "x_mean": x_mean,
                            }
                        )
                elif methodi in {"rfbbp_cox", "rfbbpcox"}:
                    from rfbbp import RFBBPCox
                    estimator = RFBBPCox(**rfbbp_class_kwargs)
                    estimator.fit(S, W, YT, Y1, Z, **rfbbp_fit_kwargs)
                    resi = estimator.summary()
                    diag = estimator.diagnostics()
                    if export_x_mean:
                        x_mean = estimator.posterior_x_mean()
                        x_mean_df = pd.DataFrame(
                            {
                                "subject_id": np.arange(x_mean.shape[0]),
                                "S": np.asarray(S).reshape(-1),
                                "W": np.asarray(W).reshape(-1),
                                "YT": np.asarray(YT).reshape(-1),
                                "Y1": np.asarray(Y1).reshape(-1),
                                "X_true": np.asarray(X).reshape(-1),
                                "x_mean": x_mean,
                            }
                        )
                elif methodi == "rfbbp_vii":
                    from rfbbp import RFBBPVIIRLS
                    estimator = RFBBPVIIRLS(**(vi_class_kwargs or {}))
                    estimator.fit(S, W, YT, Y1, Z)
                    resi = estimator.summary()
                    diag = estimator.diagnostics()
                    if export_x_mean:
                        x_mean = estimator.posterior_x_mean()
                        x_mean_df = pd.DataFrame(
                            {
                                "subject_id": np.arange(x_mean.shape[0]),
                                "S": np.asarray(S).reshape(-1),
                                "W": np.asarray(W).reshape(-1),
                                "YT": np.asarray(YT).reshape(-1),
                                "Y1": np.asarray(Y1).reshape(-1),
                                "X_true": np.asarray(X).reshape(-1),
                                "x_mean": x_mean,
                            }
                        )
                elif methodi == "rfbbp_vbn":
                    from rfbbp import RFBBPVBNewton
                    estimator = RFBBPVBNewton(**(vbn_class_kwargs or {}))
                    estimator.fit(S, W, YT, Y1, Z)
                    resi = estimator.summary()
                    diag = estimator.diagnostics()
                    if export_x_mean:
                        x_mean = estimator.posterior_x_mean()
                        x_mean_df = pd.DataFrame(
                            {
                                "subject_id": np.arange(x_mean.shape[0]),
                                "S": np.asarray(S).reshape(-1),
                                "W": np.asarray(W).reshape(-1),
                                "YT": np.asarray(YT).reshape(-1),
                                "Y1": np.asarray(Y1).reshape(-1),
                                "X_true": np.asarray(X).reshape(-1),
                                "x_mean": x_mean,
                            }
                        )
                else:
                    raise ValueError(f"Unknown method: {methodi}")
            except Exception as exc:
                print(
                    f"[WARN] {methodi} failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                resi = None
        if resi is None:
            resi = pd.DataFrame.from_records(
                [[np.nan, np.nan, np.nan, np.nan],
                 [t.interval, np.nan, np.nan, np.nan]],
                index=["beta_x", "time"],
                columns=result_columns,
            )
        elif methodi in {"rfbbp", "rfbbp_mcmc", "rfbbp_cox", "rfbbpcox", "rfbbp_vii", "rfbbp_vbn"}:
            resi = resi.loc[["beta_x"], result_columns]
            diag_numeric = pd.to_numeric(
                diag if diag is not None else pd.Series(dtype=float),
                errors="coerce",
            )
            diag_numeric.index = [
                str(idx) if str(idx).startswith("diag_") else f"diag_{idx}"
                for idx in diag_numeric.index
            ]
            diag_df = pd.DataFrame(
                {
                    "estimate": diag_numeric,
                    "post_sd": np.nan,
                    "CI_1": np.nan,
                    "CI_2": np.nan,
                }
            )
            time_df = pd.DataFrame(
                {
                    "estimate": [t.interval],
                    "post_sd": [np.nan],
                    "CI_1": [np.nan],
                    "CI_2": [np.nan],
                },
                index=["time"],
            )
            resi = pd.concat([resi, time_df, diag_df], axis=0)
        else:
            resi = np.asarray(resi, dtype=float).reshape(-1)
            resi = pd.DataFrame.from_records(
                [[resi[0], np.nan, resi[1], resi[2]],
                    [t.interval, np.nan, np.nan, np.nan],
                ],
                index=["beta_x", "time"],
                columns=result_columns,
            )
        # import ipdb; ipdb.set_trace
        res[methodi] = resi
    return res, x_mean_df


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")
    # subparsers.required = True 
    # args["subcommand"] contains the selected subcommand name

    # ============= Subcommand: generate simulated data =============
    # region
    simu_parser = subparsers.add_parser("simulate", help="generate simulated data")
    simu_parser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        help="random seed, default is 0",
    )
    simu_parser.add_argument(
        "-od",
        "--output_dir",
        default="./results/simulated_data",
        help=(
            "path to save simulated data (data.csv) and simulated "
            "parameters(params.json), default is./results/simulated_data"
        ),
    )
    simu_parser.add_argument(
        "--n_studies",
        default=4,
        type=int,
        help="number of studies, default is 4",
    )
    simu_parser.add_argument(
        "--n_samples",
        default=100,
        type=int,
        nargs="+",
        help=(
            "number of samples per study, default is 100, "
            "can be a list whose length is n_studies"
        ),
    )
    simu_parser.add_argument(
        "-bx",
        "--beta_x",
        default=0.0,
        type=float,
        help=("true beta_x, default is 0.0"),
    )
    simu_parser.add_argument(
        "--tau_beta",
        default=0.0,
        type=float,
        help=(
            "between-study SD for beta_s ~ Normal(beta_x, tau_beta^2); "
            "0 recovers the common-effect data-generating model"
        ),
    )
    simu_parser.add_argument(
        "-b0",
        "--beta_0",
        default=(-0.5, -0.25, 0.25, 0.5),
        type=float,
        nargs="+",
        help=(
            "true beta_0, default is (-0.5, -0.25, 0.25, 0.5), "
            "can be a list whose length is n_studies"
        ),
    )
    simu_parser.add_argument(
        "-a",
        "--a",
        default=(-3, 1, -1, 3),
        type=float,
        nargs="+",
        help=(
            "true a, default is (-3, 1, -1, 3), can be a list whose length is n_studies"
        ),
    )
    simu_parser.add_argument(
        "-b",
        "--b",
        default=(0.5, 0.75, 1.25, 1.5),
        type=float,
        nargs="+",
        help=(
            "true b, default is (0.5, 0.75, 1.25, 1.5), "
            "can be a list whose length is n_studies"
        ),
    )
    simu_parser.add_argument(
        "-sx",
        "--sigma2_x",
        default=1.0,
        type=float,
        help="true sigma2_x, default is 1.0",
    )
    simu_parser.add_argument(
        "-sz",
        "--sigma2_z",
        default=1.0,
        type=float,
        help="true sigma2_z, default is 1.0",
    )
    simu_parser.add_argument(
        "-se",
        "--sigma2_e",
        default=(1.0, 1.0, 1.0, 1.0),
        type=float,
        nargs="+",
        help=("true sigma2_e, default is 1.0, can be a list whose length is n_studies"),
    )
    simu_parser.add_argument(
        "-bz",
        "--beta_z",
        default=None,
        type=float,
        nargs="*",
        help=(
            "true beta_z, default is None, "
            "can be a list, the length of the list represents "
            "the number of covariates"
        ),
    )
    simu_parser.add_argument(
        "--x_distribution",
        default="normal",
        choices=("normal", "skewnorm", "lognormal"),
        help="distribution for latent X, default is normal",
    )
    simu_parser.add_argument(
        "--x_skew_shape",
        default=2.0,
        type=float,
        help="shape parameter for standardized skew-normal X, default is 5.0",
    )
    simu_parser.add_argument(
        "--x_lognorm_sigma",
        default=0.4,
        type=float,
        help=(
            "log-scale SD for standardized log-normal X. "
            "The default 0.4 gives skewness close to the observed winsorized log1p(LAR)."
        ),
    )
    simu_parser.add_argument(
        "--survival_model",
        default="auto",
        choices=("auto", "exp", "weibull", "pem", "loglogistic", "gompertz"),
        help=(
            "survival data-generating model: auto chooses exp when rho=1 "
            "and weibull otherwise; default is auto"
        ),
    )
    simu_parser.add_argument(
        "-rho",
        "--rho",
        default=1.0,
        type=float,
        help=(
            "rho, default is 1.0, can be a float scalar. "
        ),
    )
    simu_parser.add_argument(
        "-pc",
        "--p_censor",
        default=None,
        type=float,
        help=(
            "p_censor, default is None, can be a float scalar. "
        ),
    )
    simu_parser.add_argument(
        "--pem_cutpoints",
        default=(0.0, 1.0, 2.0, 3.0, 4.0),
        type=float,
        nargs="+",
        help=(
            "cutpoints for PEM truth, default is 0 1 2 3 4. "
            "The final interval is open-ended."
        ),
    )
    simu_parser.add_argument(
        "--pem_hazards",
        default=(0.05, 0.08, 0.12, 0.18, 0.25),
        type=float,
        nargs="+",
        help=(
            "baseline hazards for PEM truth. Length must equal pem_cutpoints; "
            "the last hazard applies to the open-ended final interval."
        ),
    )
    simu_parser.add_argument(
        "--loglogistic_scale",
        default=1.0,
        type=float,
        help="scale parameter for log-logistic baseline, default is 1.0",
    )
    simu_parser.add_argument(
        "--loglogistic_shape",
        default=1.5,
        type=float,
        help="shape parameter for log-logistic baseline, default is 1.5",
    )
    simu_parser.add_argument(
        "--gompertz_scale",
        default=0.08,
        type=float,
        help="scale parameter for Gompertz baseline, default is 0.08",
    )
    simu_parser.add_argument(
        "--gompertz_shape",
        default=0.25,
        type=float,
        help="shape parameter for Gompertz baseline, default is 0.25",
    )
    simu_parser.add_argument(
        "-nr",
        "--n_repeats",
        default=1000,
        type=int,
        help="number of repeats, default is 1000",
    )
    # endregion

    # ============= Subcommand: analyze simulated data =============
    # region
    ana_parser = subparsers.add_parser("analyze", help="analyze simulated data")
    ana_parser.add_argument(
        "-dd",
        "--data_dir",
        default="./results/simulated_data",
        help=("path to simulated data, default is ./results/simulated_data"),
    )
    ana_parser.add_argument(
        "-od",
        "--output_dir",
        default="./results/analyzed_results",
        help="path to save analyzed results, default is ./results/analyzed_results",
    )
    ana_parser.add_argument(
        "-m",
        "--methods",
        default=("xtrue_c", "xtrue_cs", "xtrue_ps", "wstandar", "wstandar_s", "naive", "naive_s", "rfbbp"),
        nargs="+",
        choices=("xtrue_c", "xtrue_cs", "xtrue_ps", "wstandar", "wstandar_s", "naive", "naive_s", "rfbbp", "rfbbp_mcmc", "rfbbp_cox", "rfbbpcox", "rfbbp_vii", "rfbbp_vbn"),
        help=(
            "methods to compare, default is xtrue_c, xtrue_cs, xtrue_ps, "
            "wstandar, wstandar_s, naive, naive_s, rfbbp. "
            "Use rfbbp_cox to fit RFBBP with a stratified Cox partial-likelihood survival layer."
        )
    )
    ana_parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="random seed, default is 0",
    )
    ana_parser.add_argument(
        "--cutpoints",
        default=None,
        type=float,
        nargs="*",
        help="cutpoints for piecewise exponential model, default is None",
    )
    ana_parser.add_argument(
        "--auto_cutpoints",
        action="store_true",
        help="whether to automatically generate cutpoints, default is False",
    )
    ana_parser.add_argument(
        "--study_event_cutpoint_constraint",
        action="store_true",
        help=(
            "whether to constrain automatic PEM cutpoints by the minimum "
            "number of events across studies, default is False"
        ),
    )
    ana_parser.add_argument(
        "--max_cuts",
        default=4,
        type=int,
        help="maximum number of automatically generated cutpoints, default is 4",
    )
    ana_parser.add_argument(
        "--min_events_per_interval",
        default=10,
        type=int,
        help="minimum number of events per interval when automatically generating cutpoints, default is 10",
    )
    ana_parser.add_argument(
        "--hierarchical_calibration",
        action="store_true",
        help="whether to use hierarchical calibration model, default is False",
    )
    # Prior variance
    ana_parser.add_argument(
        "--betax_sigma0",
        type=float,
        default=1.0,
        help="prior SD for beta_x, default is 1.0",
    )
    ana_parser.add_argument(
        "--a_sigma0",
        type=float,
        default=0.5,
        help="prior SD for study-specific calibration intercept a_s, default is 0.5",
    )
    ana_parser.add_argument(
        "--l_sigma0",
        type=float,
        default=0.5,
        help="prior SD for log calibration slope l_s=log(b_s), default is 0.5",
    )
    ana_parser.add_argument(
        "--sigma_w_prior",
        default="invgamma",
        type=str,
        choices=["halfnormal", "halfcauchy", "invgamma"],
        help="prior distribution for sigma_w, choices are halfnormal, halfcauchy, and invgamma; default is invgamma",
    )
    ana_parser.add_argument(
        "--sigma_w_scale0",
        type=float,
        default=1.0,
        help="scale parameter for halfnormal or halfcauchy prior on sigma_w, default is 1.0",
    )
    ana_parser.add_argument(
        "--sigma_w2_alpha0",
        type=float,
        default=3.0,
        help="alpha parameter for inverse-gamma prior on sigma_w^2, default is 3.0",
    )
    ana_parser.add_argument(
        "--sigma_w2_gamma0",
        type=float,
        default=1.0,
        help="gamma parameter for inverse-gamma prior on sigma_w^2, default is 1.0",
    )

    ana_parser.add_argument(
        "--standardize_w",
        type=str,
        choices=["none", "global", "study", "study_global"],
        default="global",
        help=(
        "How to standardize W: "
        "'none' = no standardization; "
        "'global' = global standardization using pooled mean and pooled SD; "
        "'study' = within-study standardization using study-specific mean and study-specific SD; "
        "'study_global' = within-study centering using study-specific mean and global scaling using pooled SD."
        ),
    )
    ana_parser.add_argument(
        "--draws",
        default=2000,
        type=int,
        help="number of posterior samples after tuning, default is 2000",
    )
    ana_parser.add_argument(
        "--tune",
        default=2000,
        type=int,
        help="number of tuning samples, default is 3000",
    )

    ana_parser.add_argument(
        "--chains",
        default=4,
        type=int,
        help="number of MCMC chains, default is 4",
    )

    ana_parser.add_argument(
        "--target_accept",
        default=0.9,
        type=float,
        help="target acceptance probability for NUTS sampling, default is 0.9",
    )

    ana_parser.add_argument(
        "-rfbpb",
        "--rfbbp_progressbar",
        action="store_true",
        help="whether to show progress bar when run rfbbp, default is False",
    )
    ana_parser.add_argument(
        "--vi_max_iter",
        default=500,
        type=int,
        help="maximum VI-IRLS iterations, default is 500",
    )
    ana_parser.add_argument(
        "--vi_tol",
        default=1e-6,
        type=float,
        help="VI-IRLS convergence tolerance, default is 1e-6",
    )
    ana_parser.add_argument(
        "--vi_damping",
        default=1.0,
        type=float,
        help="VI-IRLS damping factor, default is 1.0",
    )
    ana_parser.add_argument(
        "--vbn_max_iter",
        default=300,
        type=int,
        help="VB-Newton maximum iterations, default is 300",
    )
    ana_parser.add_argument(
        "--vbn_tol",
        default=1e-7,
        type=float,
        help="VB-Newton convergence tolerance, default is 1e-7",
    )
    ana_parser.add_argument(
        "--vbn_damping",
        default=1.0,
        type=float,
        help="VB-Newton damping factor, default is 1.0",
    )
    ana_parser.add_argument(
        "--export_x_mean",
        action="store_true",
        help=(
            "whether to export posterior mean of latent x for each subject "
            "from RFBBP, default is False"
        ),
    )
    ana_parser.add_argument(
        "--x_mean_output_file",
        default="rfbbp_x_mean.csv",
        help=(
            "file name for exported RFBBP latent x posterior means, "
            "default is rfbbp_x_mean.csv"
        ),
    )
    ana_parser.add_argument(
        "-nc",
        "--ncores",
        default=1,
        type=int,
        help="number of cores to use, default is 1",
    )
    # endregion

    # ============= Subcommand: compute evaluation metrics =============
    # region
    eval_parser = subparsers.add_parser(
        "evaluate", help="evaluate the performance of methods"
    )
    eval_parser.add_argument(
        "-ad",
        "--analyzed_dir",
        default="./results/analyzed_results",
        help="path to save evaluated results, default is ./results/evaluated_results",
    )
    eval_parser.add_argument(
        "-of",
        "--output_file",
        default="evaluated_results.csv",
        help="path to save evaluated results, default is evaluated_results.csv",
    )
    eval_parser.add_argument(
        "-ocf",
        "--output_ci_file",
        default="betax_ci_results.csv",
        help="path to save betax_ci results, default is betax_ci_results.csv",
    )
    eval_parser.add_argument(
        "-odf",
        "--output_diagnostic_file",
        default="model_diagnostics.csv",
        help="path to save model diagnostic results, default is model_diagnostics.csv",
    )
    # endregion

    # ============= Subcommand: summarize evaluation metrics =============
    # region
    summ_parser = subparsers.add_parser(
        "summarize", help="summarize the performance of methods"
    )
    summ_parser.add_argument(
        "-efp",
        "--evaluated_file_pattern",
        default="./results/evaluated_results/*.csv",
        help="file pattern to evaluated results, default is ./results/evaluated_results/*.csv",
    )
    summ_parser.add_argument(
        "-of",
        "--output_file",
        default="summarized_results.xlsx",
        help="path to save evaluated results, default is summarized_results.xlsx, must be a xlsx file",
    )
    summ_parser.add_argument(
        "-sp",
        "--summarize_parameters",
        type=str,
        nargs="+",
        default=("beta_x", "ratio_observed_x"),
        help="parameters to summarize, default is beta_x, ratio_observed_x",
    )

    args = parser.parse_args()

    # ================= Simulate and save data =================
    if args.subcommand == "simulate":
        if osp.exists(args.output_dir):
            raise ValueError(
                f"Output directory {args.output_dir} already "
                "exists, please remove it first."
            )

        def proc_args(x):
            if x is None:
                return x
            if isinstance(x, (list, tuple)) and len(x) == 1:
                return [x[0]] * args.n_studies
            assert len(x) == args.n_studies, (
                f"length of {x} must be equal to n_studies {args.n_studies}"
            )
            return x
                
        n_sample_per_study = proc_args(args.n_samples)

        survival_model = args.survival_model
        if survival_model == "auto":
            survival_model = "exp" if abs(args.rho - 1.0) < 1e-8 else "weibull"

        if survival_model == "exp":
            simulator = ExpSimulator(
                sigma2_e=proc_args(args.sigma2_e),
                sigma2_x=args.sigma2_x,
                sigma2_z=args.sigma2_z,
                beta0=proc_args(args.beta_0),
                betax=args.beta_x,
                tau_beta=args.tau_beta,
                betaz=proc_args(args.beta_z),
                x_distribution=args.x_distribution,
                x_skew_shape=args.x_skew_shape,
                x_lognorm_sigma=args.x_lognorm_sigma,
                a=proc_args(args.a),
                b=proc_args(args.b),
                n_sample_per_studies=n_sample_per_study,
                p_censor=args.p_censor,
            )
        elif survival_model == "weibull":
            simulator = WeibSimulator(
                sigma2_e=proc_args(args.sigma2_e),
                sigma2_x=args.sigma2_x,
                sigma2_z=args.sigma2_z,
                beta0=proc_args(args.beta_0),
                betax=args.beta_x,
                tau_beta=args.tau_beta,
                betaz=proc_args(args.beta_z),
                x_distribution=args.x_distribution,
                x_skew_shape=args.x_skew_shape,
                x_lognorm_sigma=args.x_lognorm_sigma,
                a=proc_args(args.a),
                b=proc_args(args.b),
                n_sample_per_studies=n_sample_per_study,
                rho=args.rho,
                p_censor=args.p_censor,
            )
        elif survival_model == "pem":
            simulator = PEMSimulator(
                sigma2_e=proc_args(args.sigma2_e),
                sigma2_x=args.sigma2_x,
                sigma2_z=args.sigma2_z,
                beta0=proc_args(args.beta_0),
                betax=args.beta_x,
                tau_beta=args.tau_beta,
                betaz=proc_args(args.beta_z),
                x_distribution=args.x_distribution,
                x_skew_shape=args.x_skew_shape,
                x_lognorm_sigma=args.x_lognorm_sigma,
                a=proc_args(args.a),
                b=proc_args(args.b),
                n_sample_per_studies=n_sample_per_study,
                cutpoints=args.pem_cutpoints,
                hazards=args.pem_hazards,
                p_censor=args.p_censor,
            )
        elif survival_model == "loglogistic":
            simulator = LogLogisticSimulator(
                sigma2_e=proc_args(args.sigma2_e),
                sigma2_x=args.sigma2_x,
                sigma2_z=args.sigma2_z,
                beta0=proc_args(args.beta_0),
                betax=args.beta_x,
                tau_beta=args.tau_beta,
                betaz=proc_args(args.beta_z),
                x_distribution=args.x_distribution,
                x_skew_shape=args.x_skew_shape,
                x_lognorm_sigma=args.x_lognorm_sigma,
                a=proc_args(args.a),
                b=proc_args(args.b),
                n_sample_per_studies=n_sample_per_study,
                scale=args.loglogistic_scale,
                shape=args.loglogistic_shape,
                p_censor=args.p_censor,
            )
        elif survival_model == "gompertz":
            simulator = GompertzSimulator(
                sigma2_e=proc_args(args.sigma2_e),
                sigma2_x=args.sigma2_x,
                sigma2_z=args.sigma2_z,
                beta0=proc_args(args.beta_0),
                betax=args.beta_x,
                tau_beta=args.tau_beta,
                betaz=proc_args(args.beta_z),
                x_distribution=args.x_distribution,
                x_skew_shape=args.x_skew_shape,
                x_lognorm_sigma=args.x_lognorm_sigma,
                a=proc_args(args.a),
                b=proc_args(args.b),
                n_sample_per_studies=n_sample_per_study,
                scale=args.gompertz_scale,
                shape=args.gompertz_shape,
                p_censor=args.p_censor,
            )
        else:
            raise ValueError(f"Unknown survival_model={args.survival_model!r}")
                        
        df_all = []
        for i in tqdm(range(args.n_repeats), desc="Simulate: "):
            df = simulator.simulate(seed=i + args.seed)
            df["repeat"] = i
            df_all.append(df)
        df_all = pd.concat(df_all, ignore_index=True)

        os.makedirs(args.output_dir, exist_ok=False)  # Require the output directory not to exist
        df_all.to_csv(osp.join(args.output_dir, "data.csv"), index=False)
        with open(osp.join(args.output_dir, "params.json"), "w") as f:
            args_dict = asdict(simulator)
            args_dict.update(args.__dict__)
            json.dump(
                args_dict,
                f,
                sort_keys=True,
                default=default_serializer,
            )

        return

    # ================= Load simulated data and run experiments =================
    if args.subcommand == "analyze":
        if osp.exists(args.output_dir):
            raise ValueError(
                f"Output directory {args.output_dir} already "
                "exists, please remove it first."
            )

        fn = osp.join(args.data_dir, "data.csv")
        df = pd.read_csv(fn, index_col=None)

        if "repeat" not in df.columns:
            df_iter = [(0, df)]
        else:
            df_iter = df.groupby("repeat")

        rfbbp_class_kwargs = {
            "cutpoints": args.cutpoints,
            "auto_cutpoints": args.auto_cutpoints,
            "max_cuts": args.max_cuts,
            "min_events_per_interval": args.min_events_per_interval,
            "study_event_cutpoint_constraint": args.study_event_cutpoint_constraint,
            "hierarchical_calibration": args.hierarchical_calibration,
            "a_sigma0": args.a_sigma0,
            "l_sigma0": args.l_sigma0,
            "sigma_w_prior": args.sigma_w_prior,
            "sigma_w_scale0": args.sigma_w_scale0,
            "sigma_w2_alpha0": args.sigma_w2_alpha0,
            "sigma_w2_gamma0": args.sigma_w2_gamma0,
            "standardize_w": args.standardize_w,
            "betax_sigma0": args.betax_sigma0,
        }

        rfbbp_fit_kwargs = {
            "draws": args.draws,
            "tune": args.tune,
            "chains": args.chains,
            # When repeats are distributed through mp.Pool, each worker is a
            # daemonic process and cannot spawn PyMC chain workers of its own.
            # Run the chains sequentially inside each outer worker instead.
            "cores": 1 if args.ncores > 1 else min(args.chains, 4),
            "target_accept": args.target_accept,
            "progressbar": args.rfbbp_progressbar,
            "random_seed": args.seed,
        }

        vi_class_kwargs = {
            "cutpoints": args.cutpoints,
            "max_iter": args.vi_max_iter,
            "tol": args.vi_tol,
            "damping": args.vi_damping,
            "seed": args.seed,
            "hierarchical_calibration": args.hierarchical_calibration,
            "sigma_w_prior": args.sigma_w_prior,
            "standardize_w": args.standardize_w,
            "betax_sigma0": args.betax_sigma0,
            "a_sigma0": args.a_sigma0,
            "l_sigma0": args.l_sigma0,
            "sigma_w2_alpha0": args.sigma_w2_alpha0,
            "sigma_w2_gamma0": args.sigma_w2_gamma0,
        }
        vbn_class_kwargs = {
            "cutpoints": args.cutpoints,
            "max_iter": args.vbn_max_iter,
            "tol": args.vbn_tol,
            "damping": args.vbn_damping,
            "seed": args.seed,
            "hierarchical_calibration": args.hierarchical_calibration,
            "sigma_w_prior": args.sigma_w_prior,
            "standardize_w": args.standardize_w,
            "betax_sigma0": args.betax_sigma0,
            "a_sigma0": args.a_sigma0,
            "l_sigma0": args.l_sigma0,
            "sigma_w2_alpha0": args.sigma_w2_alpha0,
            "sigma_w2_gamma0": args.sigma_w2_gamma0,
        }
        
        res_all = {k: [] for k in args.methods}
        x_mean_records = []
        
        if args.ncores <= 1:
            for i, dfi in tqdm(df_iter, desc="Analyze: "):
                zind = dfi.columns.map(lambda x: re.search(r"Z\d*", x) is not None)
                X = dfi["X_true"].values
                YT = dfi["YT"].values
                Y1 = dfi["Y1"].values
                W = dfi["W"].values
                S = dfi["S"].values
                Z = dfi.loc[:, zind].values if zind.any() else None
    
                resi, x_mean_df = analyze_data(
                    X,
                    YT,
                    Y1,
                    W,
                    S,
                    Z,
                    args.ncores,
                    args.methods,
                    rfbbp_class_kwargs,
                    rfbbp_fit_kwargs,
                    vi_class_kwargs,
                    vbn_class_kwargs,
                    args.export_x_mean,
                )

                # import ipdb;ipdb.set_trace()
                if resi is None:
                    continue
                for k, v in resi.items():
                    res_all[k].append(v)
                if x_mean_df is not None:
                    x_mean_df.insert(0, "repeat", i)
                    x_mean_records.append(x_mean_df)

        else:  # use cpu multi-processing
            with mp.Pool(
                        processes=args.ncores,
                        initializer=init_worker,
            ) as pool:
                tmp_reses = []
                for i, dfi in df_iter:
                    zind = dfi.columns.map(lambda x: re.search(r"Z\d*", x) is not None)
                    X = dfi["X_true"].values
                    YT = dfi["YT"].values
                    Y1 = dfi["Y1"].values
                    W = dfi["W"].values
                    S = dfi["S"].values
                    Z = dfi.loc[:, zind].values if zind.any() else None
                
                    tmp_resi = pool.apply_async(
                        analyze_data, 
                        (
                            X,
                            YT,
                            Y1,
                            W,
                            S,
                            Z,
                            args.ncores,
                            args.methods,
                            rfbbp_class_kwargs,
                            rfbbp_fit_kwargs,
                            vi_class_kwargs,
                            vbn_class_kwargs,
                            args.export_x_mean,
                        ),
                    )
                    if tmp_resi is None:
                        continue
                    tmp_reses.append((i, tmp_resi))
                    
                for i, tmp_resi in tqdm(tmp_reses, desc="Analyze: "):
                    resi, x_mean_df = tmp_resi.get()
                    if resi is None:
                        continue
                    for k, v in resi.items():
                        res_all[k].append(v)
                    if x_mean_df is not None:
                        x_mean_df.insert(0, "repeat", i)
                        x_mean_records.append(x_mean_df)
                        
        if all(len(res_i) == 0 for res_i in res_all.values()):
            print("unable to analyze all the data")
            os.makedirs(args.output_dir, exist_ok=False)  # Require the output directory not to exist
            empty_ds = xr.Dataset()  # Create an empty Dataset
            empty_ds.to_netcdf(osp.join(args.output_dir, "analyzed_results.nc"))
            with open(osp.join(args.output_dir, "params.json"), "w") as f:
                ana_args = args.__dict__
                json.dump(ana_args, f)
            return
        else:
            res_all = {
                k: xr.DataArray(
                    np.stack([vi for vi in v], axis=0),
                    dims=("repeat", "params", "statistic"),
                    coords={
                        "params": v[0].index.values,
                        "statistic": v[0].columns.values,
                    },
                )
            
                for k, v in res_all.items()
            }
            res_all = xr.Dataset(res_all)

            os.makedirs(args.output_dir, exist_ok=False)  # Require the output directory not to exist
            res_all.to_netcdf(osp.join(args.output_dir, "analyzed_results.nc"))
            if args.export_x_mean and x_mean_records:
                x_mean_all = pd.concat(x_mean_records, ignore_index=True)
                x_mean_all.to_csv(
                    osp.join(args.output_dir, args.x_mean_output_file),
                    index=False,
                )
            with open(osp.join(args.output_dir, "params.json"), "w") as f:
                ana_args = args.__dict__
                json.dump(ana_args, f)
            return

    # ================= Load results and parameters and compute metrics =================
    if args.subcommand == "evaluate":
        output_file = osp.join(args.analyzed_dir, args.output_file)
        if osp.exists(output_file):
            raise ValueError(
                f"Output file {output_file} already exists, please remove it first."
            )
            
        output_ci_file = osp.join(args.analyzed_dir, args.output_ci_file)
        if osp.exists(output_ci_file):
            raise ValueError(
                f"Output file {output_ci_file} already exists, please remove it first."
            )

        output_diagnostic_file = osp.join(
            args.analyzed_dir,
            args.output_diagnostic_file,
        )
        if osp.exists(output_diagnostic_file):
            raise ValueError(
                f"Output file {output_diagnostic_file} already exists, please remove it first."
            )
            
        with open(osp.join(args.analyzed_dir, "params.json"), "r") as f:
            ana_args = json.load(f)
        data_dir_resolved = resolve_simulation_data_dir(
            ana_args["data_dir"],
            args.analyzed_dir,
        )
        with open(osp.join(data_dir_resolved, "params.json"), "r") as f:
            simu_args = json.load(f)
        true_beta_x = simu_args["betax"]

        res = xr.load_dataset(osp.join(args.analyzed_dir, "analyzed_results.nc"))
        if len(res.data_vars) == 0:
            sys.exit()
        else:
            index, res_df = [], defaultdict(list)
            records = []
            diagnostic_records = []
            for k, da in res.items():
                index.append(k)

                diag_params = [
                    p for p in da.coords["params"].values
                    if str(p).startswith("diag_")
                ]
                if diag_params:
                    diag_wide = da.sel(
                        params=diag_params,
                        statistic="estimate",
                    ).to_pandas()
                    diag_wide = diag_wide.dropna(axis=0, how="all")
                    if not diag_wide.empty:
                        diag_wide.insert(0, "run_id", diag_wide.index)
                        diag_wide.insert(0, "method", k)
                        diagnostic_records.append(diag_wide.reset_index(drop=True))
                
                beta_hat = da.sel(params="beta_x", statistic="estimate").values
                ci_lower = da.sel(params="beta_x", statistic="CI_1").values
                ci_upper = da.sel(params="beta_x", statistic="CI_2").values
    
                diff = beta_hat - true_beta_x
                
                res_df["bias"].append(np.nanmean(diff))
                res_df["ab_bias"].append(np.nanmean(np.abs(diff)))
                res_df["mse"].append(np.nanmean(diff**2))
                
                n_valid = np.sum(np.isfinite(diff))
                res_df["bias_se"].append(np.nanstd(diff, ddof=1) / np.sqrt(n_valid) )
                res_df["ab_bias_se"].append(np.nanstd(np.abs(diff), ddof=1) / np.sqrt(n_valid))
                # res_df["pct_bias"].append(np.nanmean(diff)/true_beta_x*100)
                # res_df["pct_ab_bias"].append(np.nanmean(np.abs(diff))/true_beta_x*100)
                res_df["counts"].append(n_valid)
                
                in_ci = (ci_lower <= true_beta_x) & (ci_upper >= true_beta_x)
                res_df["cov_rate"].append(np.nanmean(in_ci))
                   
                time_arr = da.sel(params="time", statistic="estimate").values
                res_df["time_mean"].append(np.nanmean(time_arr))
                res_df["time_se"].append(np.nanstd(time_arr, ddof=1) / np.sqrt(np.sum(np.isfinite(time_arr))))                 

                # Add Bayesian diagnostic metrics
                empirical_sd = np.nanstd(beta_hat, ddof=1)
                mean_ci_width = np.nanmean(ci_upper - ci_lower)

                if "post_sd" in da.coords["statistic"].values:
                    post_sd = da.sel(params="beta_x", statistic="post_sd").values
                    finite_post_sd = post_sd[np.isfinite(post_sd)]
                    mean_posterior_sd = (
                        float(np.mean(finite_post_sd))
                        if finite_post_sd.size > 0
                        else np.nan
                    )
                else:
                    post_sd = np.full_like(beta_hat, np.nan, dtype=float)
                    mean_posterior_sd = np.nan

                if np.isfinite(mean_posterior_sd) and empirical_sd > 0:
                    sd_ratio = mean_posterior_sd / empirical_sd
                else:
                    sd_ratio = np.nan

                res_df["empirical_sd"].append(empirical_sd)
                res_df["mean_posterior_sd"].append(mean_posterior_sd)
                res_df["sd_ratio"].append(sd_ratio)
                res_df["mean_ci_width"].append(mean_ci_width)
                
                # Output beta_x for each replicate
                df_tmp = pd.DataFrame({
                    "method": k,
                    "run_id": range(beta_hat.shape[0]),
                    "beta_x_true": true_beta_x,
                    "beta_x_hat": beta_hat,
                    "beta_x_post_sd": post_sd,
                    "CI_lower": ci_lower,
                    "CI_upper": ci_upper,
                    "CI_width": ci_upper - ci_lower,
                    "covered": in_ci.astype(int),
                })
                records.append(df_tmp)  
                
            res_df = pd.DataFrame(res_df, index=index)
            betax_ci_all = pd.concat(records, ignore_index=True)
            print(res_df)
            res_df.to_csv(output_file)
            betax_ci_all.to_csv(output_ci_file)
            if diagnostic_records:
                diagnostics_all = pd.concat(diagnostic_records, ignore_index=True)
                diagnostics_all.to_csv(output_diagnostic_file, index=False)
            
# ================= Combine results from multiple experiments =================
    if args.subcommand == "summarize":
        all_res = []
        for fn in glob.glob(args.evaluated_file_pattern):
            path = osp.dirname(fn)
            with open(osp.join(path, "params.json"), "r") as f:
                ana_args = json.load(f)
            data_dir_resolved = resolve_simulation_data_dir(
                ana_args["data_dir"],
                path,
            )
            with open(osp.join(data_dir_resolved, "params.json"), "r") as f:
                simu_args = json.load(f)
            summ_df = pd.read_csv(fn, index_col=0)
            for k in args.summarize_parameters:
                if k not in simu_args:
                    raise ValueError(f"Parameter {k} not found in simu_args.")
                v = simu_args[k]
                if isinstance(v, (list, tuple)):
                    if len(v) == 1 or all(vi == v[0] for vi in v[1:]):
                        v = v[0]
                    else:
                        v = "-".join(map(str, v))
                summ_df[k] = v
            all_res.append(summ_df)
        all_res = pd.concat(all_res, axis=0)

        # all_res["time_se"] *= 100
        all_res["bias"] *= 100
        all_res["ab_bias"] *= 100
        all_res["mse"] *= 100
        all_res["cov_rate"] *= 100
        all_res["bias_se"] *= 100
        all_res["ab_bias_se"] *= 100

        all_res["time"] = [ 
            # f"{m:.4f}±{s:.4f}" for m, s in zip(all_res["time_mean"], all_res["time_sd"])
            f"{m:.2f}({s:.2f})"
            for m, s in zip(all_res["time_mean"], all_res["time_se"])
        ]
        all_res["bias"] = [
            # f"{m:.4f}({s:.4f})" for m, s in zip(all_res["bias"], all_res["bias_sd"])
            f"{m:.2f}({s:.2f})"
            for m, s in zip(all_res["bias"], all_res["bias_se"])
        ]
        # all_res["pct_bias"] = [
        #     f"{m:.2f}({s:.2f})"
        #     for m, s in zip(all_res["pct_bias"], all_res["bias_se"])
        # ]
        all_res["ab_bias"] = [
            # f"{m:.4f}({s:.4f})" for m, s in zip(all_res["bias"], all_res["bias_sd"])
            f"{m:.2f}({s:.2f})"
            for m, s in zip(all_res["ab_bias"], all_res["ab_bias_se"])
        ]
        # all_res["pct_ab_bias"] = [
        #     f"{m:.2f}({s:.2f})"
        #     for m, s in zip(all_res["pct_ab_bias"], all_res["ab_bias_se"])
        # ]

        # all_res.drop(columns=["time_mean", "time_sd", "bias_sd"], inplace=True)
        all_res.drop(
            columns=["time_mean", "time_se", "bias_se", "ab_bias_se"], inplace=True
        )
        all_res.index.name = "methods"
        all_res.set_index(args.summarize_parameters, append=True, inplace=True)
        all_res = all_res.unstack(level=-1).swaplevel(0, -1).swaplevel(0, -1, axis=1)
        all_res.sort_index(axis=0, inplace=True)
        all_res.sort_index(axis=1, inplace=True)
        print(all_res.to_string())
        with pd.ExcelWriter(args.output_file) as writer:
            all_res.to_excel(writer)
           
            
if __name__ == "__main__":
    main()
