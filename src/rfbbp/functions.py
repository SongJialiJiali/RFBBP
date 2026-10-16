from typing import Tuple
import numpy as np
from numpy import ndarray
from scipy.special import expit
from scipy.linalg import lstsq
from scipy.optimize import minimize
from scipy import special as sps
from time import perf_counter

def ols(
    X: ndarray, Y: ndarray, Z: ndarray | None = None
) -> Tuple[ndarray, ndarray]:
    X_des = np.stack([X, np.ones_like(X)], axis=-1)
    if Z is not None:
        X_des = np.concatenate([X_des, Z], axis=-1)

    beta, resid, _, _ = lstsq(X_des, Y)
    return beta, resid

def surv_reg(
    X: ndarray, YT: ndarray, Y1: ndarray, Z: ndarray | None = None
) -> Tuple[ndarray, ndarray]:
    eps = 1e-8  # Prevent division by zero
    beta, _ = ols(X, np.log((Y1 + eps) / (YT + eps)), Z)
    return beta

def softplus(z):
    z = np.asarray(z, dtype=float)
    az = np.abs(z)
    return np.log1p(np.exp(-az)) + np.maximum(z, 0.0)

def sigmoid(z):
    return sps.expit(z)

def transform(
    u: np.ndarray,
    use_softplus: bool,
    alpha_min: float,
    gamma_buffer: float,
    logT_max: float,
) -> tuple[float, float, float, float]:
    """Map unconstrained parameter u to alpha and gamma."""
    u_alpha, u_gamma = float(u[0]), float(u[1])
    if use_softplus:
        alpha = softplus(u_alpha) + alpha_min
        d_alpha = sigmoid(u_alpha)
        g_offset = softplus(u_gamma) + gamma_buffer
        d_gamma = sigmoid(u_gamma)
    else:
        alpha = u_alpha**2 + alpha_min
        d_alpha = 2.0 * u_alpha
        g_offset = u_gamma**2 + gamma_buffer
        d_gamma = 2.0 * u_gamma
    gamma = logT_max + g_offset
    return alpha, gamma, d_alpha, d_gamma
    
class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, *args):
        self.end = perf_counter()
        self.interval = self.end - self.start
