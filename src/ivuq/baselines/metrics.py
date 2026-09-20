"""Scoring a pricer beyond "how close is the price".

Every model, PINN or baseline, is reduced to one callable
`predict_u(m, tau) -> u` (price in strike units), and every derived quantity
below is computed from that callable by the same central finite differences.
No model gets autograd access the others lack, so the Greeks and structure
checks are a like-for-like comparison. With K = 100 the price errors are
reported in dollars on a $100 strike, matching the rest of the project.

Definitions (h = fd_m step in moneyness, k = fd_tau step in maturity):

  delta   = du/dm            ~ (u(m+h) - u(m-h)) / 2h
  gamma   = d2u/dm2          ~ (u(m+h) - 2u(m) + u(m-h)) / h^2
  u_tau   = du/dtau          ~ (u(tau+k) - u(tau-k)) / 2k

Accuracy: mae_usd, rmse_usd, max_usd against the reference price.
Greeks (European only, closed form available): delta_mae and gamma_mae, with
the true gamma taken as u'' = K * Gamma_BS since V(S) = K u(S/K).
Physics: pde_rms, the root mean square of the Black-Scholes residual
  u_tau - ((r-q) m u_m + 0.5 sigma^2 m^2 u_mm - r u)
(European only; for an American option the equation holds only in the
continuation region, so it is not reported there).

No-arbitrage structure, each a fraction of grid points that break the rule:
  viol_convex : u_mm < -tol_convex. Price is convex in the underlying.
  viol_delta  : delta outside its legal range (call [0, 1], put [-1, 0]) by tol_delta.
  viol_bound  : u below its lower bound by tol_bound. European: the discounted
                forward payoff, floored at zero. American: intrinsic value.
  viol_calendar : American only, u_tau < -tol_calendar. An American option is
                worth no less with more time to expiry.
American put, the project's main case, also gets the exercise boundary:
  boundary_mae_usd : for each of 9 maturities, the model's exercise boundary
                m*(tau) is the largest moneyness on a fine grid where the price is
                within `boundary_eps` (0.002, i.e. $0.20 on a $100 strike) of
                intrinsic value 1 - m, so u <= 1 - m + eps. The score is the mean
                |m*_model - m*_tree| over maturities, times 100 (dollars on a $100
                strike). The same rule is applied to the tree reference, so the
                detection bias is shared.
  boundary_missing : fraction of maturities where the model shows no exercise
                region at all.
An American call with q = 0 is never exercised early (it equals the European
call), so this metric is defined for puts only.

American implied volatility (the quantity traders actually quote):
  iv_mae_pts  : take the model's price at each point of a near-the-money grid
                (IV_M x IV_TAU), invert it through the 200-step CRR tree to
                get the implied vol the model's price corresponds to, and
                compare with the vol obtained the same way from the reference
                (500-step tree) price. Reported in volatility points (0.01 =
                1 vol point, times 100). Points already in the exercise
                region are dropped, since there the price does not depend on
                vol and no IV exists. The error is the price error divided by
                vega, so it punishes price errors most where the option is
                insensitive to vol (short maturity, deep in the money).
  iv_failed   : fraction of grid points where the model's price lies outside
                the range any vol in [1%, 500%] can produce, so no IV exists.
The PINNs here are trained at one fixed vol (0.20), so this measures how much
vol error each pricer's price error implies at that vol, not the ability to
fit a whole surface across vols; that needs a vol-conditioned model (planned).

Tolerances are deliberately loose (see defaults) so that ordinary fitting
noise is not counted, only genuine structural failures.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ivuq.baselines.data import Box, MarketParams
from ivuq.pricing.black_scholes import delta as bs_delta
from ivuq.pricing.black_scholes import gamma as bs_gamma
from ivuq.pricing.iv_solver import implied_vol

__all__ = ["evaluate", "exercise_boundary", "american_iv", "BOUNDARY_M", "BOUNDARY_TAU", "IV_M", "IV_TAU"]

K_REPORT = 100.0
BOUNDARY_M = np.linspace(0.4, 1.0, 121)
BOUNDARY_TAU = np.linspace(0.1, 0.9, 9)
IV_M = np.linspace(0.9, 1.2, 7)
IV_TAU = np.array([0.25, 0.5, 0.75])


def exercise_boundary(predict_u: Callable[[np.ndarray, np.ndarray], np.ndarray], eps: float = 2e-3) -> np.ndarray:
    """Exercise boundary m*(tau) of an American put at each BOUNDARY_TAU, nan
    where the price never comes within eps of intrinsic value."""
    mm, tt = np.meshgrid(BOUNDARY_M, BOUNDARY_TAU)
    u = np.asarray(predict_u(mm.ravel(), tt.ravel())).reshape(mm.shape)
    in_exercise = u <= (1.0 - mm) + eps
    out = np.full(len(BOUNDARY_TAU), np.nan)
    for i, row in enumerate(in_exercise):
        if row.any():
            out[i] = BOUNDARY_M[np.nonzero(row)[0].max()]
    return out


def american_iv(
    predict_u: Callable[[np.ndarray, np.ndarray], np.ndarray],
    params: MarketParams,
    steps: int = 200,
    active: np.ndarray | None = None,
) -> np.ndarray:
    """Implied vol of an American put implied by predict_u at each IV_M x IV_TAU
    point (flattened, meshgrid order); nan where inversion is impossible or,
    if `active` is given, where active is False."""
    mm, tt = np.meshgrid(IV_M, IV_TAU)
    m, tau = mm.ravel(), tt.ravel()
    u = np.asarray(predict_u(m, tau), dtype=np.float64)
    out = np.full(m.shape, np.nan)
    for i in range(len(m)):
        if active is not None and not active[i]:
            continue
        try:
            out[i] = implied_vol(float(u[i]), float(m[i]), 1.0, float(tau[i]), params.r, params.q, "put", True, steps=steps)
        except ValueError:
            pass
    return out


def evaluate(
    predict_u: Callable[[np.ndarray, np.ndarray], np.ndarray],
    m: np.ndarray,
    tau: np.ndarray,
    u_ref: np.ndarray,
    *,
    style: str,
    option_type: str,
    params: MarketParams,
    train_box: Box | None = None,
    boundary_ref: np.ndarray | None = None,
    iv_ref: np.ndarray | None = None,
    fd_m: float = 0.03,
    fd_tau: float = 0.02,
    tol_convex: float = 0.05,
    tol_delta: float = 0.02,
    tol_bound: float = 5e-4,
    tol_calendar: float = 1e-3,
) -> dict[str, float]:
    m = np.asarray(m, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    u = predict_u(m, tau)
    err = u - u_ref

    out: dict[str, float] = {
        "mae_usd": K_REPORT * float(np.mean(np.abs(err))),
        "rmse_usd": K_REPORT * float(np.sqrt(np.mean(err**2))),
        "max_usd": K_REPORT * float(np.max(np.abs(err))),
    }
    if train_box is not None:
        inside = train_box.contains(m, tau)
        out["mae_inside_usd"] = K_REPORT * float(np.mean(np.abs(err[inside]))) if inside.any() else float("nan")
        out["mae_outside_usd"] = K_REPORT * float(np.mean(np.abs(err[~inside]))) if (~inside).any() else float("nan")

    u_p, u_m = predict_u(m + fd_m, tau), predict_u(m - fd_m, tau)
    delta = (u_p - u_m) / (2 * fd_m)
    gamma = (u_p - 2 * u + u_m) / fd_m**2
    u_tau = (predict_u(m, tau + fd_tau) - predict_u(m, tau - fd_tau)) / (2 * fd_tau)

    r, q, sigma = params.r, params.q, params.sigma
    is_call = option_type == "call"

    if style == "european":
        d_true = np.array([bs_delta(mi * K_REPORT, K_REPORT, ti, r, q, sigma, option_type) for mi, ti in zip(m, tau)])
        g_true = K_REPORT * np.array([bs_gamma(mi * K_REPORT, K_REPORT, ti, r, q, sigma) for mi, ti in zip(m, tau)])
        out["delta_mae"] = float(np.mean(np.abs(delta - d_true)))
        out["gamma_mae"] = float(np.mean(np.abs(gamma - g_true)))
        resid = u_tau - ((r - q) * m * delta + 0.5 * sigma**2 * m**2 * gamma - r * u)
        out["pde_rms"] = float(np.sqrt(np.mean(resid**2)))
        forward = m * np.exp(-q * tau) - np.exp(-r * tau)
        lower = np.maximum(forward if is_call else -forward, 0.0)
    else:
        out["delta_mae"] = out["gamma_mae"] = out["pde_rms"] = float("nan")
        lower = np.maximum(m - 1.0, 0.0) if is_call else np.maximum(1.0 - m, 0.0)

    out["boundary_mae_usd"] = out["boundary_missing"] = float("nan")
    if style == "american" and option_type == "put" and boundary_ref is not None:
        b = exercise_boundary(predict_u)
        out["boundary_missing"] = float(np.mean(np.isnan(b)))
        if not np.all(np.isnan(b)):
            out["boundary_mae_usd"] = K_REPORT * float(np.nanmean(np.abs(b - boundary_ref)))

    out["iv_mae_pts"] = out["iv_failed"] = float("nan")
    if style == "american" and option_type == "put" and iv_ref is not None:
        active = np.isfinite(iv_ref)
        iv = american_iv(predict_u, params, active=active)
        out["iv_failed"] = float(np.mean(np.isnan(iv[active])))
        ok = active & np.isfinite(iv)
        if ok.any():
            out["iv_mae_pts"] = 100.0 * float(np.mean(np.abs(iv[ok] - iv_ref[ok])))

    out["viol_convex"] = float(np.mean(gamma < -tol_convex))
    lo_d, hi_d = (0.0, 1.0) if is_call else (-1.0, 0.0)
    out["viol_delta"] = float(np.mean((delta < lo_d - tol_delta) | (delta > hi_d + tol_delta)))
    out["viol_bound"] = float(np.mean(u < lower - tol_bound))
    out["viol_calendar"] = float(np.mean(u_tau < -tol_calendar)) if style == "american" else float("nan")
    return out
