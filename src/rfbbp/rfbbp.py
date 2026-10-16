from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import logging
import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import pytensor.tensor as pt
import arviz as az

from rfbbp.rfbbp_common import (
    prepare_piecewise_survival_data,
    prepare_rfbbp_data,
)

import warnings

warnings.filterwarnings(
    "ignore",
    message="PyTensor could not link to a BLAS installation.*"
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="overflow encountered in dot",
    module=r"pymc\.step_methods\.hmc\.quadpotential",
)

def _as_1d_array(x, name: str, dtype=None) -> np.ndarray:
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape={arr.shape}.")
    return arr


def make_piecewise_survival_data(
    YT: np.ndarray,
    Y1: np.ndarray,
    cutpoints: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subject, interval, exposure, event, _ = prepare_piecewise_survival_data(
        YT=YT,
        Y1=Y1,
        cutpoints=cutpoints,
    )
    return subject, interval, exposure, event

@dataclass 
class RFBBPPyMC:
    """
    Reference-Free Bayesian Biomarker Pooling using PyMC.

    The class intentionally mimics a simple estimator interface:

        model = RFBBPPyMC(cutpoints=[0, 3, 6, 12, 24])
        model.fit(S=S, W=W, YT=YT, Y1=Y1, Z=Z)
        print(model.summary())
        print(model.hr_summary())
    """

    cutpoints: Optional[Sequence[float]] = None
    auto_cutpoints: bool = True
    max_cuts: int = 2
    min_events_per_interval: int = 10
    study_event_cutpoint_constraint: bool = False

    # Priors for survival model
    betax_mu0: float = 0.0
    betax_sigma0: float = 1.0
    betaz_mu0: float = 0.0
    betaz_sigma0: float = 1.0

    # Random-walk baseline prior
    beta0_first_mu0: float = 0.0
    beta0_first_sigma0: float = 1.0
    sigma_delta_alpha0: float = 3.0
    sigma_delta_gamma0: float = 0.5

    # Calibration model priors
    hierarchical_calibration: bool = False
    mu_a_mu0: float = 0.0
    mu_a_sigma0: float = 1.0
    tau_a_sigma0: float = 1.0
    mu_l_mu0: float = 0.0
    mu_l_sigma0: float = 0.5
    tau_l_sigma0: float = 0.5

    a_mu0: float = 0.0
    a_sigma0: float = 0.5
    l_mu0: float = 0.0
    l_sigma0: float = 0.5

    # Measurement error prior: sigma_w_s > 0 
    sigma_w_prior: Literal["halfnormal", "halfcauchy", "invgamma"] = "invgamma"
    sigma_w_scale0: float = 1.0
    sigma_w2_alpha0: float = 3.0
    sigma_w2_gamma0: float = 1.0

    # Sampling controls
    solver: Literal["pymc", "numpyro", "blackjax"] = "pymc"
    draws: int = 2000 
    tune: int = 2000 
    chains: int = 4 
    mcmc_cores: int = 4
    target_accept: float = 0.9 
    seed: int = 123
    pbar: bool = False 
    
    standardize_w: Literal["none", "global", "study", "study_global"] = "none"

    def _make_cutpoints(
        self,
        YT: np.ndarray,
        Y1: np.ndarray,
        study_idx: Optional[np.ndarray] = None,
        use_study_event_constraint: bool = False,
    ) -> Optional[list[float]]:

        event_times = np.sort(YT[Y1 == 1])
        if event_times.size == 0:
            return None

        max_intervals = event_times.size // self.min_events_per_interval
        
        if use_study_event_constraint and study_idx is not None:
            study_event_counts = [
                np.sum((study_idx == s) & (Y1 == 1))
                for s in np.unique(study_idx)
            ]
            max_intervals_by_study = (
                min(study_event_counts) // self.min_events_per_interval
            )
            max_intervals = min(max_intervals, max_intervals_by_study)

        if max_intervals < 2:
            return None

        max_allowed_intervals = self.max_cuts + 1
        n_intervals_start = min(max_allowed_intervals, max_intervals)

        for n_intervals in range(n_intervals_start, 1, -1):
            n_cuts = n_intervals - 1
            probs = np.linspace(0, 1, n_cuts + 2)[1:-1]
            internal_cuts = np.unique(np.quantile(event_times, probs))

            cutpoints = np.r_[0.0, internal_cuts, YT.max() + 1e-8]
            cutpoints = np.unique(cutpoints)

            actual_intervals = cutpoints.size - 1
            if actual_intervals < 2:
                continue

            ok = True
            for k in range(actual_intervals):
                left = cutpoints[k]
                right = cutpoints[k + 1]
                if k == actual_intervals - 1:
                    in_interval = (YT >= left) & (YT <= right) & (Y1 == 1)
                else:
                    in_interval = (YT >= left) & (YT < right) & (Y1 == 1)

                if np.sum(in_interval) < self.min_events_per_interval:
                    ok = False
                    break

            if not ok:
                continue

            if use_study_event_constraint and study_idx is not None:
                for s in np.unique(study_idx):
                    for k in range(actual_intervals):
                        left = cutpoints[k]
                        right = cutpoints[k + 1]
                        if k == actual_intervals - 1:
                            in_interval_s = (
                                (study_idx == s)
                                & (YT >= left)
                                & (YT <= right)
                                & (Y1 == 1)
                            )
                        else:
                            in_interval_s = (
                                (study_idx == s)
                                & (YT >= left)
                                & (YT < right)
                                & (Y1 == 1)
                            )

                        if np.sum(in_interval_s) < self.min_events_per_interval:
                            ok = False
                            break

                    if not ok:
                        break

            if ok:
                return cutpoints.tolist()

        return None
    
    def _prepare_data(
        self,
        S: np.ndarray,
        W: np.ndarray,
        YT: np.ndarray,
        Y1: np.ndarray,
        Z: Optional[np.ndarray] = None,
    ) -> None:
        S_arr = np.asarray(S)
        YT_arr = np.asarray(YT, dtype=float)
        Y1_arr = np.asarray(Y1, dtype=int)
        studies_tmp, study_idx_tmp = np.unique(S_arr, return_inverse=True)

        if self.cutpoints is None and self.auto_cutpoints:
            # self.cutpoints_ = self._make_cutpoints(YT, Y1)
            self.cutpoints_ = self._make_cutpoints(
                YT=YT_arr,
                Y1=Y1_arr,
                study_idx=study_idx_tmp,
                use_study_event_constraint=self.study_event_cutpoint_constraint,
            )
        else:
            self.cutpoints_ = self.cutpoints

        data = prepare_rfbbp_data(
            S=S,
            W=W,
            YT=YT,
            Y1=Y1,
            Z=Z,
            cutpoints=self.cutpoints_,
            standardize_w=self.standardize_w,
        )

        self.data_ = data
        self.S_raw_ = data.S_raw 
        self.studies_ = data.studies 
        self.study_idx_ = data.study_idx 
        self.W_ = data.W
        self.W_original_ = data.W_original
        if "W_mean" in data.standardization:
            self.W_mean_ = data.standardization["W_mean"]
        if "W_sd" in data.standardization:
            self.W_sd_ = data.standardization["W_sd"]
        self.YT_ = data.YT
        self.Y1_ = data.Y1
        self.Z_ = data.Z

        self.subject_long_ = data.subject_long 
        self.study_long_ = data.study_long 
        self.interval_long_ = data.interval_long
        self.exposure_long_ = data.exposure_long 
        self.event_long_ = data.event_long 

        self.n_ = data.n
        self.n_studies_ = data.n_studies 
        self.n_intervals_ = data.n_intervals 
        self.n_z_ = data.n_z 

    def _sigma_w_prior(self, n_studies: int):
        if self.sigma_w_prior == "halfnormal":
            return pm.HalfNormal("sigma_w", sigma=self.sigma_w_scale0, shape=n_studies)
        if self.sigma_w_prior == "halfcauchy":
            return pm.HalfCauchy("sigma_w", beta=self.sigma_w_scale0, shape=n_studies)
        if self.sigma_w_prior == "invgamma":
            sigma_w2 = pm.InverseGamma(
                "sigma_w2",
                alpha=self.sigma_w2_alpha0,
                beta=self.sigma_w2_gamma0,
                shape=n_studies,
            )
            return pm.Deterministic("sigma_w", pt.sqrt(sigma_w2))
        raise ValueError(f"Unknown sigma_w_prior={self.sigma_w_prior!r}")

    def _build_model(self) -> pm.Model:
        S = self.n_studies_
        K = self.n_intervals_ 
        pz = self.n_z_ 


        with pm.Model() as model:
            if self.hierarchical_calibration:
                mu_a = pm.Normal("mu_a", mu=self.mu_a_mu0, sigma=self.mu_a_sigma0)
                tau_a = pm.HalfNormal("tau_a", sigma=self.tau_a_sigma0)
                a_raw = pm.Normal("a_raw", mu=0.0, sigma=1.0, shape=S)
                a = pm.Deterministic("a", mu_a + tau_a * a_raw) 

                mu_l = pm.Normal("mu_l", mu=self.mu_l_mu0, sigma=self.mu_l_sigma0)
                tau_l = pm.HalfNormal("tau_l", sigma=self.tau_l_sigma0)
                l_raw = pm.Normal("l_raw", mu=0.0, sigma=1.0, shape=S)
                l = pm.Deterministic("l", mu_l + tau_l * l_raw)
            else:
                a = pm.Normal("a", mu=self.a_mu0, sigma=self.a_sigma0, shape=S)
                l = pm.Normal("l", mu=self.l_mu0, sigma=self.l_sigma0, shape=S)
            b = pm.Deterministic("b", pt.exp(l))
            sigma_w = self._sigma_w_prior(S)

            a_i = a[self.study_idx_]
            b_i = b[self.study_idx_]
            sigma_i = sigma_w[self.study_idx_]

            total_var_i = b_i**2 + sigma_i**2
            total_sd_i = pt.sqrt(total_var_i)
            pm.Normal("W_obs", mu=a_i, sigma=total_sd_i, observed=self.W_)

            x_cond_mean = b_i * (self.W_ - a_i) / total_var_i
            x_cond_sd = sigma_i / total_sd_i
            z_x = pm.Normal("z_x", mu=0.0, sigma=1.0, shape=self.n_)
            x = pm.Deterministic("x", x_cond_mean + x_cond_sd * z_x)

            sigma_delta2 = pm.InverseGamma(
                "sigma_delta2",
                alpha=self.sigma_delta_alpha0,
                beta=self.sigma_delta_gamma0,
            )
            sigma_delta = pm.Deterministic("sigma_delta", pt.sqrt(sigma_delta2))
            beta0_first = pm.Normal(
                "beta0_first",
                mu=self.beta0_first_mu0,
                sigma=self.beta0_first_sigma0,
                shape=S,
            )

            if K == 1:
                beta0 = pm.Deterministic("beta0", beta0_first[:, None])

            else:
                # beta0_increments = pm.Normal(
                #     "beta0_increments",
                #     mu=0.0,
                #     sigma=sigma_delta,
                #     shape=(S, K - 1),
                # )
                beta0_increments_raw = pm.Normal(
                    "beta0_increments_raw",
                    mu=0.0,
                    sigma=1.0,
                    shape=(S, K - 1),
                )
                beta0_increments = pm.Deterministic(
                    "beta0_increments",
                    sigma_delta * beta0_increments_raw,
                )
                beta0_rest = beta0_first[:, None] + pt.cumsum(beta0_increments, axis=1)
                beta0 = pm.Deterministic(
                    "beta0",
                    pt.concatenate([beta0_first[:, None], beta0_rest], axis=1),
                )

            beta_x = pm.Normal(
                "beta_x",
                mu=self.betax_mu0,
                sigma=self.betax_sigma0,
            )

            if pz > 0:
                beta_z = pm.Normal(
                    "beta_z",
                    mu=self.betaz_mu0,
                    sigma=self.betaz_sigma0,
                    shape=pz,
                )
                z_part = pt.dot(self.Z_[self.subject_long_, :], beta_z)
            else:
                z_part = 0.0

            # ηik
            eta_long = (
                beta0[self.study_long_, self.interval_long_]
                + beta_x * x[self.subject_long_]
                + z_part
            )
            # μ=t*exp(η)
            mu_long = self.exposure_long_ * pt.exp(eta_long)
            pm.Poisson(
                "d_obs",
                mu=mu_long,
                observed=self.event_long_,
            )
            # HR
            pm.Deterministic("HR_x", pt.exp(beta_x))

        return model

    def fit(
        self,
        S: np.ndarray,
        W: np.ndarray,
        YT: np.ndarray,
        Y1: np.ndarray,
        Z: Optional[np.ndarray] = None,
        **sample_kwargs,
    ) -> "RFBBPPyMC":
        import time

        start = time.perf_counter()
        self._prepare_data(S=S, W=W, YT=YT, Y1=Y1, Z=Z)
        self.model_ = self._build_model()
        if not self.pbar:
            logging.getLogger("pymc").setLevel(logging.ERROR)
            logging.getLogger("pymc.sampling").setLevel(logging.ERROR)
            logging.getLogger("pymc.sampling.mcmc").setLevel(logging.ERROR)
        with self.model_:
            kwargs = dict(
                draws=self.draws,
                tune=self.tune,
                chains=self.chains,
                cores=self.mcmc_cores,
                target_accept=self.target_accept,
                random_seed=self.seed,
                progressbar=self.pbar,
                compute_convergence_checks=self.pbar,
                init="adapt_diag",
            )
            kwargs.update(sample_kwargs)
            if self.solver in ("numpyro", "blackjax"):
                kwargs["nuts_sampler"] = self.solver
            self.idata_ = pm.sample(**kwargs)

        self.runtime_seconds_ = time.perf_counter() - start
        return self

    def summary(
        self,
        var_names: Optional[Sequence[str]] = ("beta_x", "HR_x", "a", "b", "sigma_w"),
        hdi_prob: float = 0.95,
    ) -> pd.DataFrame:
        summ = az.summary(self.idata_, var_names=list(var_names), hdi_prob=hdi_prob, kind="stats")
        hdi_cols = [col for col in summ.columns if col.startswith("hdi_")]
        out = summ[["mean", "sd"] + hdi_cols].copy()
        out.columns = ["estimate", "post_sd", "CI_1", "CI_2"]

        return out

    def diagnostics(
        self,
        var_names: Optional[Sequence[str]] = (
            "beta_x",
            "HR_x",
            "a",
            "b",
            "sigma_w",
            "beta0",
            "sigma_delta",
        ),
    ) -> pd.Series:
        """Return compact MCMC diagnostics and fitted model dimensions."""
        existing_var_names = [
            name for name in var_names
            if name in self.idata_.posterior.data_vars
        ]
        diagnostics = {
            "diag_n": float(self.n_),
            "diag_n_studies": float(self.n_studies_),
            "diag_n_intervals": float(self.n_intervals_),
            "diag_n_events": float(np.sum(self.Y1_)),
            "diag_n_cutpoints": float(
                0 if self.cutpoints_ is None else len(self.cutpoints_)
            ),
        }

        if existing_var_names:
            summ = az.summary(
                self.idata_,
                var_names=existing_var_names,
                kind="diagnostics",
            )
            for col in ("r_hat", "ess_bulk", "ess_tail"):
                vals = summ[col].to_numpy(dtype=float) if col in summ.columns else []
                vals = vals[np.isfinite(vals)]
                if col == "r_hat":
                    diagnostics["diag_max_r_hat"] = (
                        float(np.max(vals)) if vals.size else np.nan
                    )
                elif col == "ess_bulk":
                    diagnostics["diag_min_ess_bulk"] = (
                        float(np.min(vals)) if vals.size else np.nan
                    )
                elif col == "ess_tail":
                    diagnostics["diag_min_ess_tail"] = (
                        float(np.min(vals)) if vals.size else np.nan
                    )
            if "beta_x" in self.idata_.posterior.data_vars:
                beta_x_diag = az.summary(
                    self.idata_,
                    var_names=["beta_x"],
                    kind="diagnostics",
                )
                row = beta_x_diag.iloc[0]
                diagnostics["diag_beta_x_r_hat"] = float(row.get("r_hat", np.nan))
                diagnostics["diag_beta_x_ess_bulk"] = float(row.get("ess_bulk", np.nan))
                diagnostics["diag_beta_x_ess_tail"] = float(row.get("ess_tail", np.nan))
        else:
            diagnostics["diag_max_r_hat"] = np.nan
            diagnostics["diag_min_ess_bulk"] = np.nan
            diagnostics["diag_min_ess_tail"] = np.nan

        sample_stats = self.idata_.sample_stats
        n_draws = float(np.prod(list(sample_stats.sizes.values())))
        if "diverging" in sample_stats:
            n_divergent = float(sample_stats["diverging"].sum().item())
            diagnostics["diag_n_divergent"] = n_divergent
            diagnostics["diag_divergence_rate"] = n_divergent / n_draws
        if "tree_depth" in sample_stats:
            tree_depth = sample_stats["tree_depth"].values
            max_tree_depth = float(np.nanmax(tree_depth))
            diagnostics["diag_max_tree_depth"] = max_tree_depth
            diagnostics["diag_n_max_tree_depth"] = float(
                np.sum(tree_depth == max_tree_depth)
            )
        if "acceptance_rate" in sample_stats:
            diagnostics["diag_mean_acceptance_rate"] = float(
                sample_stats["acceptance_rate"].mean().item()
            )
        if "step_size" in sample_stats:
            diagnostics["diag_mean_step_size"] = float(
                sample_stats["step_size"].mean().item()
            )
        elif "step_size_bar" in sample_stats:
            diagnostics["diag_mean_step_size"] = float(
                sample_stats["step_size_bar"].mean().item()
            )
        if "energy" in sample_stats:
            diagnostics["diag_mean_energy"] = float(
                sample_stats["energy"].mean().item()
            )
            try:
                bfmi = np.asarray(az.bfmi(self.idata_), dtype=float)
                for i, value in enumerate(bfmi):
                    diagnostics[f"diag_bfmi_chain_{i}"] = float(value)
                diagnostics["diag_min_bfmi"] = float(np.nanmin(bfmi))
            except Exception:
                diagnostics["diag_min_bfmi"] = np.nan

        diagnostics["rhat"] = diagnostics.get("diag_max_r_hat", np.nan)
        diagnostics["ess_bulk"] = diagnostics.get("diag_min_ess_bulk", np.nan)
        diagnostics["ess_tail"] = diagnostics.get("diag_min_ess_tail", np.nan)
        diagnostics["divergences"] = diagnostics.get("diag_n_divergent", 0.0)
        diagnostics["runtime_seconds"] = float(getattr(self, "runtime_seconds_", np.nan))

        return pd.Series(diagnostics, dtype=float)

    def posterior_x_mean(self) -> np.ndarray:
        return (
            self.idata_.posterior["x"]
            .mean(dim=("chain", "draw"))
            .values
        )


if __name__ == "__main__":
    df = pd.read_csv("test.csv")

    S = df["S"].to_numpy()
    W = df["W"].to_numpy()
    YT = df["YT"].to_numpy()
    Y1 = df["Y1"].to_numpy()

    Z_cols = []
    Z = None if len(Z_cols) == 0 else df[Z_cols].to_numpy()

    model = RFBBPPyMC(
        cutpoints=None,          # exponential model
        # cutpoints=[0, 1, 2, 3, 4, 5, 6],  # piecewise exponential model
        auto_cutpoints=False,
        betax_sigma0=1,
        # betaz_sigma0=1.0,
        beta0_first_sigma0 = 1.0,
        # sigma_delta_alpha0 = 3.0,
        # sigma_delta_gamma0 = 0.5,
        hierarchical_calibration=False,
        # mu_a_sigma0 = 1.0,
        # tau_a_sigma0 = 1.0,
        # mu_l_sigma0 = 0.5,
        # tau_l_sigma0 = 0.5,
        a_sigma0 = 1,
        l_sigma0 = 0.5,
        # sigma_w_prior="halfcauchy",
        # sigma_w_prior="halfnormal",
        sigma_w_prior="invgamma",
        # sigma_w_scale0 = 1.0,
        sigma_w2_alpha0 = 3.0,
        sigma_w2_gamma0 = 1.0,
        
        solver="pymc",
        draws=2000,
        tune=2000,
        chains=4,
        mcmc_cores=4,
        target_accept=0.9,
        seed=666,
        pbar=True,
        # standardize_w="global",
        # standardize_w="none",
        standardize_w="study",
        # standardize_w= "study_global",
    )

    model.fit(S, W, YT, Y1, Z)
    print(model.summary(var_names=("beta_x", "HR_x","a", "b", "sigma_w")))
#     print(az.summary(
#     model.idata_,
#     var_names=["beta_x", "HR_x", "a", "b", "sigma_w"],
#     kind="all"
# ))
#     az.plot_pair(
#     model.idata_,
#     var_names=["b", "sigma_w"],
#     divergences=True)
#     plt.savefig("pair_plot_divergences.png", dpi=300, bbox_inches="tight")
#     plt.close()
