"""Split conformal prediction on a PointPredictor.

Depends only on the PointPredictor protocol. Does not import torch or a PINN
class. kappa = 0 is ordinary split conformal (Lei et al., 2018). kappa > 0
divides the residual by a difficulty weight; for PINNs that weight is
|PDE residual|.

Mondrian conformal computes a separate quantile in each bucket (moneyness
and tenor by default).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "IntervalResult",
    "conformal_quantile",
    "auto_kappa",
    "nonconformity_scores",
    "split_conformal_interval",
    "mondrian_split_conformal_interval",
    "moneyness_tenor_buckets",
    "empirical_coverage",
    "mean_width",
]


@dataclass(frozen=True)
class IntervalResult:
    """Intervals in the same units as `u` (strike units if K=1)."""

    u_hat: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    q: float | dict[int, float]
    kappa: float
    alpha: float
    n_cal: int


def _as_1d(*arrays: np.ndarray) -> list[np.ndarray]:
    out = [np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays]
    n = out[0].shape[0]
    for a in out[1:]:
        if a.shape[0] != n:
            raise ValueError("all arrays must have the same length")
    return out


def _difficulty(predictor, m: np.ndarray, tau: np.ndarray) -> np.ndarray:
    r = predictor.constraint_residual(m, tau)
    if r is None:
        return np.zeros_like(m, dtype=np.float64)
    return np.abs(np.asarray(r, dtype=np.float64).reshape(-1))


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile.

    q is the k-th order statistic of `scores` with k = ceil((n+1)(1-alpha)).
    If k > n the quantile is +inf (the interval must be the whole line to
    keep the coverage guarantee).
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    s = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    n = s.size
    if n == 0:
        raise ValueError("need at least one calibration score")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(s[k - 1])


def auto_kappa(difficulty: np.ndarray) -> float:
    """Scale so a typical |R| is order one: kappa = 1 / median(|R|)."""
    d = np.asarray(difficulty, dtype=np.float64).reshape(-1)
    med = float(np.median(d))
    if med <= 0.0:
        return 0.0
    return 1.0 / med


def nonconformity_scores(
    u: np.ndarray,
    u_hat: np.ndarray,
    difficulty: np.ndarray,
    kappa: float,
) -> np.ndarray:
    if kappa < 0.0:
        raise ValueError(f"kappa must be >= 0, got {kappa}")
    denom = 1.0 + kappa * np.asarray(difficulty, dtype=np.float64).reshape(-1)
    return np.abs(np.asarray(u, dtype=np.float64).reshape(-1) - np.asarray(u_hat, dtype=np.float64).reshape(-1)) / denom


def split_conformal_interval(
    predictor,
    m_cal: np.ndarray,
    tau_cal: np.ndarray,
    u_cal: np.ndarray,
    m_test: np.ndarray,
    tau_test: np.ndarray,
    *,
    alpha: float = 0.1,
    kappa: float = 0.0,
    scale_kappa: bool = False,
) -> IntervalResult:
    """Split conformal intervals for a fitted PointPredictor.

    `u_cal` are calibration labels in the same units as `predictor.predict`
    (u = V/K). Returned `lo`/`hi` are in those units. Multiply by K for
    dollars. `scale_kappa=True` ignores `kappa` and sets it from the
    calibration residuals so median |R| * kappa = 1.
    """
    m_cal, tau_cal, u_cal = _as_1d(m_cal, tau_cal, u_cal)
    m_test, tau_test = _as_1d(m_test, tau_test)

    u_hat_cal = np.asarray(predictor.predict(m_cal, tau_cal), dtype=np.float64).reshape(-1)
    d_cal = _difficulty(predictor, m_cal, tau_cal)
    if scale_kappa:
        kappa = auto_kappa(d_cal)
    scores = nonconformity_scores(u_cal, u_hat_cal, d_cal, kappa)
    q = conformal_quantile(scores, alpha)

    u_hat = np.asarray(predictor.predict(m_test, tau_test), dtype=np.float64).reshape(-1)
    d_test = _difficulty(predictor, m_test, tau_test)
    half = q * (1.0 + kappa * d_test)
    return IntervalResult(
        u_hat=u_hat,
        lo=u_hat - half,
        hi=u_hat + half,
        q=q,
        kappa=float(kappa),
        alpha=float(alpha),
        n_cal=int(u_cal.size),
    )


def moneyness_tenor_buckets(m: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Six buckets: {m < 0.9, 0.9-1.1, m > 1.1} x {tau < 0.4, tau >= 0.4}."""
    m = np.asarray(m, dtype=np.float64).reshape(-1)
    tau = np.asarray(tau, dtype=np.float64).reshape(-1)
    m_bin = np.where(m < 0.9, 0, np.where(m <= 1.1, 1, 2))
    t_bin = np.where(tau < 0.4, 0, 1)
    return (m_bin * 2 + t_bin).astype(np.int64)


def mondrian_split_conformal_interval(
    predictor,
    m_cal: np.ndarray,
    tau_cal: np.ndarray,
    u_cal: np.ndarray,
    m_test: np.ndarray,
    tau_test: np.ndarray,
    *,
    alpha: float = 0.1,
    kappa: float = 0.0,
    scale_kappa: bool = False,
    buckets_cal: np.ndarray | None = None,
    buckets_test: np.ndarray | None = None,
) -> IntervalResult:
    """One conformal quantile per bucket. Empty test buckets get NaN intervals."""
    m_cal, tau_cal, u_cal = _as_1d(m_cal, tau_cal, u_cal)
    m_test, tau_test = _as_1d(m_test, tau_test)
    if buckets_cal is None:
        buckets_cal = moneyness_tenor_buckets(m_cal, tau_cal)
    if buckets_test is None:
        buckets_test = moneyness_tenor_buckets(m_test, tau_test)
    buckets_cal = np.asarray(buckets_cal).reshape(-1)
    buckets_test = np.asarray(buckets_test).reshape(-1)
    if buckets_cal.shape[0] != m_cal.shape[0] or buckets_test.shape[0] != m_test.shape[0]:
        raise ValueError("bucket arrays must match calibration / test length")

    u_hat_cal = np.asarray(predictor.predict(m_cal, tau_cal), dtype=np.float64).reshape(-1)
    d_cal = _difficulty(predictor, m_cal, tau_cal)
    if scale_kappa:
        kappa = auto_kappa(d_cal)
    scores = nonconformity_scores(u_cal, u_hat_cal, d_cal, kappa)

    q_by_bucket: dict[int, float] = {}
    for b in np.unique(buckets_cal):
        q_by_bucket[int(b)] = conformal_quantile(scores[buckets_cal == b], alpha)

    u_hat = np.asarray(predictor.predict(m_test, tau_test), dtype=np.float64).reshape(-1)
    d_test = _difficulty(predictor, m_test, tau_test)
    q_test = np.empty(m_test.shape[0], dtype=np.float64)
    for i, b in enumerate(buckets_test):
        b = int(b)
        q_test[i] = q_by_bucket[b] if b in q_by_bucket else np.nan
    half = q_test * (1.0 + kappa * d_test)
    return IntervalResult(
        u_hat=u_hat,
        lo=u_hat - half,
        hi=u_hat + half,
        q=q_by_bucket,
        kappa=float(kappa),
        alpha=float(alpha),
        n_cal=int(u_cal.size),
    )


def empirical_coverage(lo: np.ndarray, hi: np.ndarray, u: np.ndarray) -> float:
    lo, hi, u = _as_1d(lo, hi, u)
    return float(np.mean((u >= lo) & (u <= hi)))


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    lo, hi = _as_1d(lo, hi)
    return float(np.mean(hi - lo))
