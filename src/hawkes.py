import warnings
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution


class HawkesProcess:
    # univariate Hawkes with exponential kernel
    # λ*(t) = μ + Σ_{t_i < t} α * exp(-β * (t - t_i))

    def __init__(self, mu: float = 1.0, alpha: float = 0.5, beta: float = 2.0):
        if mu <= 0:   raise ValueError("mu must be > 0")
        if beta <= 0: raise ValueError("beta must be > 0")
        self.mu    = float(mu)
        self.alpha = float(alpha)
        self.beta  = float(beta)

    def simulate(self, T: float, seed: int = None) -> np.ndarray:
        # Ogata thinning
        rng      = np.random.default_rng(seed)
        times    = []
        t        = 0.0
        lam_star = self.mu

        while t < T:
            dt = rng.exponential(1.0 / lam_star)
            t += dt
            if t > T: break
            lam_t = self.intensity(t, np.array(times))
            if rng.uniform() <= lam_t / lam_star:
                times.append(t)
                lam_star = lam_t + self.alpha
            else:
                lam_star = lam_t

        return np.array(times)

    def intensity(self, t: float, times: np.ndarray) -> float:
        past = times[times < t]
        return self.mu + self.alpha * np.sum(np.exp(-self.beta * (t - past)))

    def intensity_path(self, times: np.ndarray, T: float, n_grid: int = 500) -> Tuple[np.ndarray, np.ndarray]:
        grid = np.linspace(0.0, T, n_grid)
        lam  = np.array([self.intensity(t, times) for t in grid])
        return grid, lam

    def log_likelihood(self, times: np.ndarray, T: float) -> float:
        # recursive trick from Ozaki (1979) -- avoids recomputing the full sum at each step
        if len(times) == 0:
            return -self.mu * T
        R = np.zeros(len(times))
        for k in range(1, len(times)):
            R[k] = np.exp(-self.beta * (times[k] - times[k-1])) * (1.0 + R[k-1])
        lam_star = self.mu + self.alpha * R
        if np.any(lam_star <= 0): return -np.inf
        integral = self.mu * T + (self.alpha / self.beta) * np.sum(1 - np.exp(-self.beta * (T - times)))
        return float(np.sum(np.log(lam_star)) - integral)

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    @property
    def stationary_intensity(self) -> float:
        n = self.branching_ratio
        if n >= 1: raise ValueError("process not stationary (alpha/beta >= 1)")
        return self.mu / (1.0 - n)

    def __repr__(self) -> str:
        return f"HawkesProcess(mu={self.mu:.4f}, alpha={self.alpha:.4f}, beta={self.beta:.4f}, n={self.branching_ratio:.3f})"


@dataclass
class CalibrationResult:
    mu:             float
    alpha:          float
    beta:           float
    log_likelihood: float
    n_events:       int
    T:              float
    success:        bool
    message:        str

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    @property
    def process(self) -> HawkesProcess:
        return HawkesProcess(self.mu, self.alpha, self.beta)

    def __repr__(self) -> str:
        return (f"CalibrationResult\n"
                f"  mu={self.mu:.6f}  alpha={self.alpha:.6f}  beta={self.beta:.6f}\n"
                f"  n=alpha/beta={self.branching_ratio:.4f}\n"
                f"  log_likelihood={self.log_likelihood:.4f}  N={self.n_events}  T={self.T:.1f}\n"
                f"  success={self.success}")


def calibrate(times: np.ndarray, T: float, method: str = "L-BFGS-B",
              n_restarts: int = 5, verbose: bool = False) -> CalibrationResult:
    times = np.sort(np.asarray(times, dtype=float))
    times = times[times <= T]
    n     = len(times)

    if n < 5:
        warnings.warn("few events — estimates may be unreliable")

    lam_hat = max(n / T, 1e-6)
    # mu bounded by empirical rate, alpha < 1 (stationarity), beta à la louche
    bounds  = [(1e-6, 10 * lam_hat), (1e-6, 0.999), (1e-4, 100.0)]

    def neg_ll(params):
        mu, alpha, beta = params
        if mu <= 0 or alpha < 0 or beta <= 0: return 1e10
        ll = HawkesProcess(mu, alpha, beta).log_likelihood(times, T)
        return -ll if np.isfinite(ll) else 1e10

    best_res, best_val = None, np.inf

    if method == "differential_evolution":
        res = differential_evolution(neg_ll, bounds, seed=0, maxiter=600, tol=1e-9)
        best_res, best_val = res, res.fun
    else:
        rng = np.random.default_rng(0)
        for trial in range(n_restarts):
            x0 = np.array([rng.uniform(1e-6, 5*lam_hat), rng.uniform(0.05, 0.80), rng.uniform(0.5, 20.0)])
            try:
                res = minimize(neg_ll, x0, method=method, bounds=bounds,
                               options={"maxiter": 2000, "ftol": 1e-12})
                if verbose:
                    print(f"  restart {trial}: LL={-res.fun:.4f}  success={res.success}")
                if res.fun < best_val:
                    best_val, best_res = res.fun, res
            except Exception as e:
                if verbose: print(f"  restart {trial} failed: {e}")

    if best_res is None:
        return CalibrationResult(np.nan, np.nan, np.nan, -np.inf, n, T, False, "optimization failed")

    mu_h, alpha_h, beta_h = best_res.x
    return CalibrationResult(
        mu=float(mu_h), alpha=float(alpha_h), beta=float(beta_h),
        log_likelihood=float(-best_val), n_events=n, T=T,
        success=bool(best_res.success), message=best_res.message)


def moment_estimator(times: np.ndarray, T: float) -> Tuple[float, float, float]:
    # pas parfait mais ça marche pour initialiser le MLE
    times = np.sort(np.asarray(times, dtype=float))
    n     = len(times)
    if n < 10:
        return float(n / T), 0.3, 2.0
    lam_hat  = n / T
    dt       = np.diff(times)
    cv2      = dt.var() / (dt.mean() ** 2)
    n_hat    = float(np.clip(1.0 - 1.0 / cv2, 0.0, 0.95)) if cv2 > 1 else 0.3
    mu_hat   = lam_hat * (1.0 - n_hat)
    beta_hat = max(2.0 / dt.mean(), 0.5)
    return mu_hat, n_hat * beta_hat, beta_hat
