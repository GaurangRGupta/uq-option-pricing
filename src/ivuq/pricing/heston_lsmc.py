"""Longstaff-Schwartz Monte Carlo (Longstaff & Schwartz, 2001, Review of
Financial Studies): the American-option reference pricer under Heston.

No closed form exists for an American option under any dynamics (early
exercise makes it a free-boundary problem, not a plain PDE), and under
Heston there isn't even a fast lattice method the way CRR is for GBM --
so LSMC plays the same validation role here that `binomial.crr_price`
plays for N2/GBM: the classical, independently-checkable answer that
`AmericanHestonPINN` (N2-Heston) is validated against. Standard numerical-
finance machinery, same category as `heston.py`'s closed-form European
price and `binomial.py`'s tree -- unrelated to and predating every paper in
`planning/papers/PAPER_TRAIL.md`.

Path simulation. Heston's variance follows a square-root (CIR) diffusion,
which can go negative under a naive Euler step. Uses "full truncation"
Euler (Lord, Koekkoek & Van Dijk, 2010): the drift and diffusion of v both
use max(v, 0) in place of v, and v itself is allowed to go negative between
steps but gets clamped back to 0 before the next step's diffusion term is
evaluated. Simpler than the QE scheme and not as tight for large steps, but
correct and easy to verify -- same "correctness first" choice `binomial.py`
makes over speed.

Regression (the Longstaff-Schwartz step). At each exercise date, working
backward from maturity, regress the (discounted) cash flow each in-the-money
path would receive under continuation onto a polynomial basis of that path's
current (moneyness m, variance v):

    [1, m, m^2, v, v^2, m*v]

against ordinary least squares. Compares the fitted continuation value to
immediate exercise; where exercise is worth more, that path's cash flow is
overwritten with the intrinsic value at this date (Longstaff & Schwartz's
own recipe, applied here with a small polynomial basis in place of their
Laguerre-in-S basis, extended to the second state variable v the same way
multi-factor LSMC applications generally do -- there's nothing Heston-
specific about the regression step itself, only the basis functions it's
given).
"""

from __future__ import annotations

import numpy as np

__all__ = ["lsmc_american_heston_price"]


def _simulate_paths(
    S0: float,
    v0: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    S = np.empty((n_paths, n_steps + 1))
    v = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    v[:, 0] = v0

    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rng.standard_normal(n_paths)
        w_v = z1
        w_s = rho * z1 + np.sqrt(1.0 - rho**2) * z2

        v_pos = np.maximum(v[:, t], 0.0)
        v[:, t + 1] = v[:, t] + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * sqrt_dt * w_v
        S[:, t + 1] = S[:, t] * np.exp((r - q - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * w_s)

    return S, v


def lsmc_american_heston_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    v0: float,
    option_type: str,
    n_steps: int = 100,
    n_paths: int = 20000,
    seed: int = 0,
) -> float:
    """American option price under Heston via Longstaff-Schwartz Monte
    Carlo. `n_steps` doubles as the Bermudan-exercise-date count that
    approximates true continuous exercise (same discretization role
    `steps` plays in `binomial.crr_price`) and the path time-discretization
    count."""
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if T <= 0:
        return max(S0 - K, 0.0) if option_type == "call" else max(K - S0, 0.0)

    rng = np.random.default_rng(seed)
    S, v = _simulate_paths(S0, v0, r, q, kappa, theta, xi, rho, T, n_steps, n_paths, rng)
    dt = T / n_steps
    discount = np.exp(-r * dt)

    if option_type == "call":
        payoff = np.maximum(S - K, 0.0)
    else:
        payoff = np.maximum(K - S, 0.0)

    m = S / K  # basis in moneyness units, same convention as the PDE code
    cashflow = payoff[:, -1].copy()

    for t in range(n_steps - 1, 0, -1):
        cashflow = cashflow * discount
        itm = payoff[:, t] > 0.0
        if np.any(itm):
            m_t, v_t = m[itm, t], v[itm, t]
            X = np.column_stack([np.ones(itm.sum()), m_t, m_t**2, v_t, v_t**2, m_t * v_t])
            coeffs, *_ = np.linalg.lstsq(X, cashflow[itm], rcond=None)
            continuation = X @ coeffs
            exercise = payoff[itm, t]
            exercise_now = exercise > continuation
            cashflow[np.where(itm)[0][exercise_now]] = exercise[exercise_now]

    cashflow = cashflow * discount
    return float(np.mean(cashflow))
