"""Implied volatility inversion: European (closed-form BS) and American (tree).

American IV is solved by bracketing crr_price() with Brent's method. BAW
is used to sanity-check the bracket, not yet to narrow it — the speed
optimizations in BUILD_PLAN.md 6.3 (numba, BAW warm-start, caching, adaptive
step count, joblib) are a Phase-8 concern for running this across a whole
dataset. This module is correct-and-simple first.
"""

from __future__ import annotations

from scipy.optimize import brentq

from ivuq.pricing import black_scholes
from ivuq.pricing.binomial import crr_price

__all__ = ["implied_vol"]


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    is_american: bool,
    steps: int = 500,
    lo: float = 1e-2,
    hi: float = 5.0,
) -> float:
    """Invert either the Black-Scholes formula (European) or the CRR tree (American).

    `lo` defaults to 1% volatility rather than near-zero: at very low sigma the CRR
    tree's risk-neutral probability can fall outside (0, 1) for realistic step counts
    (u - d shrinks faster than growth - d as sigma -> 0), which is a discretization
    artifact, not a real price. No traded option prices in below 1% IV in practice,
    so this bound costs nothing.
    """
    if not is_american:
        return black_scholes.implied_vol(market_price, S, K, T, r, q, option_type, lo=lo, hi=hi)

    def f(sigma: float) -> float:
        return crr_price(S, K, T, r, q, sigma, option_type, is_american=True, steps=steps) - market_price

    lo_val, hi_val = f(lo), f(hi)
    if lo_val > 0 or hi_val < 0:
        raise ValueError(
            f"market_price={market_price} is outside the reachable American price range "
            f"for sigma in [{lo}, {hi}] (bracket: [{lo_val + market_price}, {hi_val + market_price}])"
        )
    return brentq(f, lo, hi, xtol=1e-6, rtol=1e-8, maxiter=100)
