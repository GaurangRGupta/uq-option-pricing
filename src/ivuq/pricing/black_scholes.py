"""Black-Scholes-Merton pricing, Greeks, and European implied volatility.

Conventions (see DECISIONS.md):
  - Continuously compounded rate `r` and dividend yield `q`.
  - `T` is time to expiry in years, ACT/365.
  - `sigma` is annualized volatility.
  - option_type is "call" or "put".
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

__all__ = [
    "d1",
    "d2",
    "price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "implied_vol",
]


def _validate_option_type(option_type: str) -> str:
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return option_type


def d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be strictly positive")
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    return d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)


def price(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str) -> float:
    """European option price under Black-Scholes-Merton with continuous dividend yield q."""
    option_type = _validate_option_type(option_type)
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic
    _d1 = d1(S, K, T, r, q, sigma)
    _d2 = _d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * np.exp(-q * T) * norm.cdf(-_d1)


def delta(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str) -> float:
    option_type = _validate_option_type(option_type)
    _d1 = d1(S, K, T, r, q, sigma)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(_d1)
    return -np.exp(-q * T) * norm.cdf(-_d1)


def gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    _d1 = d1(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(_d1) / (S * sigma * np.sqrt(T))


def vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Sensitivity to a 1.0 (100 vol point) change in sigma. Divide by 100 for a 1-vol-point vega."""
    _d1 = d1(S, K, T, r, q, sigma)
    return S * np.exp(-q * T) * norm.pdf(_d1) * np.sqrt(T)


def theta(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str) -> float:
    """Per-year theta (divide by 365 for per-day)."""
    option_type = _validate_option_type(option_type)
    _d1 = d1(S, K, T, r, q, sigma)
    _d2 = _d1 - sigma * np.sqrt(T)
    term1 = -S * np.exp(-q * T) * norm.pdf(_d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        return term1 - r * K * np.exp(-r * T) * norm.cdf(_d2) + q * S * np.exp(-q * T) * norm.cdf(_d1)
    return term1 + r * K * np.exp(-r * T) * norm.cdf(-_d2) - q * S * np.exp(-q * T) * norm.cdf(-_d1)


def rho(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str) -> float:
    """Sensitivity to a 1.0 (100%) change in r. Divide by 100 for a 1% rho."""
    option_type = _validate_option_type(option_type)
    _d2 = d2(S, K, T, r, q, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(_d2)
    return -K * T * np.exp(-r * T) * norm.cdf(-_d2)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    lo: float = 1e-6,
    hi: float = 5.0,
) -> float:
    """Invert the Black-Scholes formula for volatility via Brent's method.

    Raises ValueError if `market_price` sits outside the model's no-arbitrage
    bounds for any sigma in [lo, hi], since no root can exist there.
    """
    option_type = _validate_option_type(option_type)

    def f(sigma: float) -> float:
        return price(S, K, T, r, q, sigma, option_type) - market_price

    lo_val, hi_val = f(lo), f(hi)
    if lo_val > 0 or hi_val < 0:
        raise ValueError(
            f"market_price={market_price} is outside the reachable Black-Scholes "
            f"price range for sigma in [{lo}, {hi}] (bracket: [{lo_val + market_price}, "
            f"{hi_val + market_price}])"
        )
    return brentq(f, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=200)
