import logging
import os
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats

logger = logging.getLogger(__name__)

def default_serializer(obj):
    """Custom serializer for objects not serializable by default json encoder."""
    if isinstance(obj, np.ndarray):
        # Convert ndarray to a list and then to a tuple to ensure hashability
        # and consistent ordering for the purpose of generating a unique name.
        # Using a tuple of its shape and a tuple of its flattened elements
        # ensures that both shape and content are considered.
        return (obj.shape, tuple(obj.flatten().tolist()))
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

@dataclass
class Simulator:
    """the base class for simulating data from a bayesian biomarker pooling model"""

    mu_x: float = 0
    sigma2_x: float = 1.0
    betax: float = 1.0
    tau_beta: float = 0.0
    beta0: Union[float, Sequence[float]] = 1.0
    a: Sequence[float] = (-3, 1, -1, 3)
    b: Sequence[float] = (0.5, 0.75, 1.25, 1.5)
    sigma2_e: Union[float, Sequence[float]] = 1.0
    n_sample_per_studies: Union[int, Sequence[int]] = 100
    direction: Literal["x->w", "w->x"] = "x->w"
    mu_z: float = 0
    sigma2_z: float = 1
    betaz: Sequence[float] | None = None
    x_distribution: Literal["normal", "skewnorm", "lognormal"] = "normal"
    x_skew_shape: float = 2.0
    x_lognorm_sigma: float = 0.4

    def param_names_by_studies(self) -> list[str]:
        return [
            "beta0",
            "a",
            "b",
            "sigma2_e",
            "n_sample_per_studies",
        ]

    def __post_init__(self):
        # logger = logging.getLogger("main.simulate")

        assert self.direction in ["x->w", "w->x"], "direction must be x->w or w->x"
        if self.tau_beta < 0:
            raise ValueError("tau_beta must be non-negative.")

        ns = None
        for param_name in self.param_names_by_studies():
            parami = getattr(self, param_name)
            if isinstance(parami, (int, float)):
                continue
            len_parami = len(parami)
            if ns is None:
                ns = len_parami
            else:
                assert ns == len_parami, (
                    "the length of a, b, sigma2_e, beta0, beta1 must be one or equal."
                )
        assert ns is not None, (
            "the number of studies can not be identified by simulation settings."
        )
        for param_name in self.param_names_by_studies():
            parami = getattr(self, param_name)
            setattr(
                self,
                param_name,
                np.array([parami] * ns)
                if isinstance(parami, (float, int))
                else np.array(parami),
            )

        if self.direction == "w->x":
            self.mu_w = (self.mu_x - self.a) / self.b
            self.sigma2_w = (self.sigma2_x - self.sigma2_e) / (self.b**2)

    @property
    def name(self) -> str:
        serialized_dict = json.dumps(
            asdict(self), sort_keys=True, default=default_serializer
        )
        # Create a SHA-256 hash object
        hasher = hashlib.sha256()
        # Update the hasher with the serialized dictionary string (encoded to bytes)
        hasher.update(serialized_dict.encode("utf-8"))
        # Get the hexadecimal representation of the hash
        return hasher.hexdigest()

    def generate_Y(self, rng, X, W, Y_, Z) -> np.ndarray:
        raise NotImplementedError

    def generate_group(self, rng, X, W, Y, Z) -> np.ndarray:
        # Start at 1
        return np.repeat(
            np.arange(1, len(self.n_sample_per_studies) + 1),
            self.n_sample_per_studies,
        )

    def simulate(self, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        a = np.repeat(self.a, self.n_sample_per_studies)
        b = np.repeat(self.b, self.n_sample_per_studies)
        sigma_e = np.repeat(self.sigma2_e**0.5, self.n_sample_per_studies)
        e = rng.normal(0, sigma_e)
        if self.direction == "w->x":
            mu_w = np.repeat(self.mu_w, self.n_sample_per_studies)
            sigma_w = np.repeat(self.sigma2_w**0.5, self.n_sample_per_studies)
            W = rng.normal(mu_w, sigma_w)
            X = a + b * W + e
        else:  # x->w
            n_total = sum(self.n_sample_per_studies)
            if self.x_distribution == "normal":
                X = rng.normal(self.mu_x, self.sigma2_x**0.5, size=n_total)
            elif self.x_distribution == "skewnorm":
                X_raw = stats.skewnorm.rvs(
                    self.x_skew_shape,
                    size=n_total,
                    random_state=rng,
                )
                delta = self.x_skew_shape / np.sqrt(1 + self.x_skew_shape**2)
                x_mean = delta * np.sqrt(2 / np.pi)
                x_sd = np.sqrt(1 - 2 * delta**2 / np.pi)
                X = self.mu_x + self.sigma2_x**0.5 * (X_raw - x_mean) / x_sd
            elif self.x_distribution == "lognormal":
                X_raw = rng.lognormal(mean=0.0, sigma=self.x_lognorm_sigma, size=n_total)
                x_mean = np.exp(self.x_lognorm_sigma**2 / 2)
                x_sd = np.sqrt(
                    (np.exp(self.x_lognorm_sigma**2) - 1)
                    * np.exp(self.x_lognorm_sigma**2)
                )
                X = self.mu_x + self.sigma2_x**0.5 * (X_raw - x_mean) / x_sd
            else:
                raise ValueError(f"Unknown x_distribution={self.x_distribution!r}")
            W = a + b * X + e

        if self.betaz is not None:
            Z = rng.normal(
                self.mu_z,
                self.sigma2_z**0.5,
                size=(sum(self.n_sample_per_studies), len(self.betaz)),
            )
        else:
            Z = None

        beta0 = np.repeat(self.beta0, self.n_sample_per_studies)
        # Optional between-study exposure-effect heterogeneity.  Avoid drawing
        # when tau_beta=0 so historical simulations retain their exact RNG
        # stream and remain reproducible.
        if self.tau_beta > 0:
            beta_by_study = rng.normal(self.betax, self.tau_beta, size=len(self.n_sample_per_studies))
        else:
            beta_by_study = np.full(len(self.n_sample_per_studies), self.betax, dtype=float)
        beta_by_subject = np.repeat(beta_by_study, self.n_sample_per_studies)
        Y_ = beta_by_subject * X + beta0

        if self.betaz is not None:
            Y_ += np.dot(Z, self.betaz)

        Y = self.generate_Y(rng, X, W, Y_, Z)
        group = self.generate_group(rng, X, W, Y, Z)

        X_obs = X.copy()

        res = pd.DataFrame(
            {
                "W": W,
                "X_true": X,
                "YT": Y[:,0],
                "Y1": Y[:,1],
                "S": group,
                "beta_s_true": beta_by_subject,
            }
        )
        if self.betaz is not None:
            for i in range(len(self.betaz)):
                res[f"Z{i + 1}"] = Z[:, i]
        return res

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, sort_keys=True, default=default_serializer)
            
@dataclass
class ExpSimulator(Simulator):
    """Simulator for survival outcomes with exponential distribution."""

    p_censor: float = 0.3            # Target censoring rate

    def _solve_tau(self, lam):
        """Solve for tau using the target censoring rate: P(T > C) = p_censor."""
        lam = np.maximum(lam, 1e-12)
        def censor_rate(tau):
            return np.mean((1 - np.exp(-lam * tau)) / (lam * tau)) 
        
        res = optimize.root_scalar(
        lambda tau: censor_rate(tau) - self.p_censor,
        bracket=[1e-8, 1e4],    # Search bracket
        method="brentq"         # Root-finding method
    )
        return res.root

    def generate_Y(self, rng, X, W, Y_, Z) -> np.ndarray:
        lam = np.exp(Y_)  
        U = rng.uniform(size=len(Y_))
        T = -np.log(U) / lam

        tau = self._solve_tau(lam)
        C = rng.uniform(0, tau, size=len(Y_))

        delta = (T <= C).astype(int)
        Y = np.column_stack((np.minimum(T, C), delta)) 
        return Y
   
    
@dataclass
class WeibSimulator(Simulator):
    """Simulator for survival outcomes with weibull distribution."""

    rho: float = 1.5
    p_censor: float = 0.3            # Target censoring rate

    def _solve_tau(self, lam):
        lam = np.maximum(lam, 1e-12)
        def censor_rate(tau):
            x = (lam * tau) ** self.rho                    # (n,)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                lower = special.gammainc(1/self.rho, x) * special.gamma(1/self.rho)
                val = lower / (tau * self.rho * lam)
            # Avoid NaN or infinite values under extreme inputs
            val = np.where(np.isfinite(val), val, 0.0)
            return float(np.mean(val))
        
        res = optimize.root_scalar(
            lambda tau: censor_rate(tau) - self.p_censor,
            bracket=[1e-8, 1e4], 
            method="brentq"
            )
        return res.root

    def generate_Y(self, rng, X, W, Y_, Z) -> np.ndarray:
        lam = np.exp(Y_ / self.rho)  
        U = rng.uniform(size=len(Y_))
        T = (-np.log(U) ) ** (1.0 / self.rho) / lam

        tau = self._solve_tau(lam)
        C = rng.uniform(0, tau, size=len(Y_))

        delta = (T <= C).astype(int)
        Y = np.column_stack((np.minimum(T, C), delta)) 
        return Y


@dataclass
class PEMSimulator(Simulator):
    """Simulator for survival outcomes with a piecewise exponential baseline."""

    cutpoints: Sequence[float] = (0, 1, 2, 3, 4)
    hazards: Sequence[float] = (0.05, 0.08, 0.12, 0.18, 0.25)
    p_censor: float = 0.3

    def __post_init__(self):
        super().__post_init__()
        self.cutpoints = np.asarray(self.cutpoints, dtype=float)
        self.hazards = np.asarray(self.hazards, dtype=float)
        if self.cutpoints.ndim != 1 or self.hazards.ndim != 1:
            raise ValueError("cutpoints and hazards must be one-dimensional.")
        if self.cutpoints.size < 1:
            raise ValueError("cutpoints must contain at least the first boundary.")
        if not np.isclose(self.cutpoints[0], 0.0):
            raise ValueError("cutpoints[0] must be 0.")
        if np.any(np.diff(self.cutpoints) <= 0):
            raise ValueError("cutpoints must be strictly increasing.")
        if self.hazards.size != self.cutpoints.size:
            raise ValueError(
                "hazards length must equal cutpoints length. The last hazard "
                "applies to the open-ended final interval."
            )
        if np.any(self.hazards <= 0):
            raise ValueError("hazards must be positive.")

    def _cumhaz0(self, t):
        t = np.asarray(t, dtype=float)
        res = np.zeros_like(t, dtype=float)
        for k, lam in enumerate(self.hazards):
            left = self.cutpoints[k]
            right = self.cutpoints[k + 1] if k + 1 < self.cutpoints.size else np.inf
            width = np.maximum(0.0, np.minimum(t, right) - left)
            res += lam * width
        return res

    def _invert_cumhaz0(self, h):
        h = np.asarray(h, dtype=float)
        t = np.zeros_like(h, dtype=float)
        remaining = h.copy()
        active = np.ones_like(h, dtype=bool)
        for k, lam in enumerate(self.hazards):
            left = self.cutpoints[k]
            if k + 1 < self.cutpoints.size:
                width = self.cutpoints[k + 1] - left
                interval_h = lam * width
                in_interval = active & (remaining <= interval_h)
                t[in_interval] = left + remaining[in_interval] / lam
                remaining[active & ~in_interval] -= interval_h
                active &= ~in_interval
            else:
                t[active] = left + remaining[active] / lam
                active[:] = False
        return t

    def _survival_integral(self, tau, risk):
        total = np.zeros_like(risk, dtype=float)
        prev_cumhaz0 = 0.0
        for k, lam in enumerate(self.hazards):
            left = self.cutpoints[k]
            right = self.cutpoints[k + 1] if k + 1 < self.cutpoints.size else tau
            if tau <= left:
                break
            upper = min(tau, right)
            width = upper - left
            if width <= 0:
                continue
            rate = risk * lam
            total += np.exp(-risk * prev_cumhaz0) * (-np.expm1(-rate * width)) / rate
            prev_cumhaz0 += lam * width
            if tau <= right:
                break
        return total

    def _solve_tau(self, risk):
        risk = np.maximum(risk, 1e-12)

        def censor_rate(tau):
            return float(np.mean(self._survival_integral(tau, risk) / tau))

        upper = 1.0
        while censor_rate(upper) > self.p_censor:
            upper *= 2.0
            if upper > 1e8:
                raise RuntimeError("Unable to bracket tau for target censoring rate.")

        res = optimize.root_scalar(
            lambda tau: censor_rate(tau) - self.p_censor,
            bracket=[1e-8, upper],
            method="brentq",
        )
        return res.root

    def generate_Y(self, rng, X, W, Y_, Z) -> np.ndarray:
        risk = np.exp(Y_)
        U = rng.uniform(size=len(Y_))
        target_cumhaz0 = -np.log(U) / risk
        T = self._invert_cumhaz0(target_cumhaz0)

        tau = self._solve_tau(risk)
        C = rng.uniform(0, tau, size=len(Y_))

        delta = (T <= C).astype(int)
        Y = np.column_stack((np.minimum(T, C), delta))
        return Y


@dataclass
class LogLogisticSimulator(Simulator):
    """Simulator with proportional-hazards log-logistic baseline."""

    scale: float = 1.0
    shape: float = 2.0
    p_censor: float = 0.3

    def _cumhaz0(self, t):
        t = np.asarray(t, dtype=float)
        return np.log1p((self.scale * np.maximum(t, 0.0)) ** self.shape)

    def _invert_cumhaz0(self, h):
        h = np.asarray(h, dtype=float)
        return np.expm1(h) ** (1.0 / self.shape) / self.scale

    def _solve_tau(self, risk):
        risk = np.maximum(risk, 1e-12)

        def censor_rate(tau):
            grid = np.linspace(0.0, tau, 256)
            surv = np.exp(-risk[:, None] * self._cumhaz0(grid)[None, :])
            return float(np.trapezoid(surv.mean(axis=0), grid) / tau)

        upper = 1.0
        while censor_rate(upper) > self.p_censor:
            upper *= 2.0
            if upper > 1e8:
                raise RuntimeError("Unable to bracket tau for target censoring rate.")

        res = optimize.root_scalar(
            lambda tau: censor_rate(tau) - self.p_censor,
            bracket=[1e-8, upper],
            method="brentq",
        )
        return res.root

    def generate_Y(self, rng, X, W, Y_, Z) -> np.ndarray:
        risk = np.exp(Y_)
        U = rng.uniform(size=len(Y_))
        target_cumhaz0 = -np.log(U) / risk
        T = self._invert_cumhaz0(target_cumhaz0)

        tau = self._solve_tau(risk)
        C = rng.uniform(0, tau, size=len(Y_))

        delta = (T <= C).astype(int)
        Y = np.column_stack((np.minimum(T, C), delta))
        return Y


@dataclass
class GompertzSimulator(Simulator):
    """Simulator with proportional-hazards Gompertz baseline."""

    scale: float = 0.08
    shape: float = 0.25
    p_censor: float = 0.3

    def _cumhaz0(self, t):
        t = np.asarray(t, dtype=float)
        t = np.maximum(t, 0.0)
        if abs(self.shape) < 1e-10:
            return self.scale * t
        return self.scale * np.expm1(self.shape * t) / self.shape

    def _invert_cumhaz0(self, h):
        h = np.asarray(h, dtype=float)
        if abs(self.shape) < 1e-10:
            return h / self.scale
        return np.log1p(h * self.shape / self.scale) / self.shape

    def _solve_tau(self, risk):
        risk = np.maximum(risk, 1e-12)

        def censor_rate(tau):
            grid = np.linspace(0.0, tau, 256)
            surv = np.exp(-risk[:, None] * self._cumhaz0(grid)[None, :])
            return float(np.trapezoid(surv.mean(axis=0), grid) / tau)

        upper = 1.0
        while censor_rate(upper) > self.p_censor:
            upper *= 2.0
            if upper > 1e8:
                raise RuntimeError("Unable to bracket tau for target censoring rate.")

        res = optimize.root_scalar(
            lambda tau: censor_rate(tau) - self.p_censor,
            bracket=[1e-8, upper],
            method="brentq",
        )
        return res.root

    def generate_Y(self, rng, X, W, Y_, Z) -> np.ndarray:
        risk = np.exp(Y_)
        U = rng.uniform(size=len(Y_))
        target_cumhaz0 = -np.log(U) / risk
        T = self._invert_cumhaz0(target_cumhaz0)

        tau = self._solve_tau(risk)
        C = rng.uniform(0, tau, size=len(Y_))

        delta = (T <= C).astype(int)
        Y = np.column_stack((np.minimum(T, C), delta))
        return Y
    
if __name__ == "__main__":
    simulator = WeibSimulator(
                sigma2_x=1.0,
                beta0=[-0.5, -0.25, 0.25, 0.5],
                a=[-3, 1, -1, 3],
                b=[0.5, 0.75, 1.25, 1.5],
                sigma2_e=[1.0, 1.0, 1.0, 1.0],
                n_sample_per_studies=100,
                betaz=None,
                betax=0.9,
                rho=1.5,
                p_censor=0.3,
            )
    df = simulator.simulate(seed=1)
    df.to_csv("test.csv", index=False)
    print(df.head())
