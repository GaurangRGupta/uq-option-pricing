"""Barone-Adesi-Whaley (1987) quadratic approximation for American options.

Fast closed-form-ish approximation used to warm-start the binomial-tree
implied-vol solver (BUILD_PLAN.md 6.3). The binomial tree, not this module,
is the reference American pricer — BAW only needs to be close, not exact.

The Newton-Raphson step for the early-exercise critical price is derived
directly from calculus here (not copied from a reference implementation),
because several widely-circulated transcriptions of the original VBA pseudocode
conflate the normal CDF and PDF in the derivative term. Verified against the
binomial tree in tests/test_baw.py.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from ivuq.pricing.black_scholes import price as bs_price

__all__ = ["baw_price"]

_MAX_ITER = 100
_TOL = 1e-8


def _d1(S: float, K: float, T: float, b: float, sigma: float) -> float:
    return (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _q2(K: float, T: float, r: float, b: float, sigma: float) -> float:
    M = 2 * r / sigma**2
    N = 2 * b / sigma**2
    K_disc = 1 - np.exp(-r * T)
    return (-(N - 1) + np.sqrt((N - 1) ** 2 + 4 * M / K_disc)) / 2


def _q1(K: float, T: float, r: float, b: float, sigma: float) -> float:
    M = 2 * r / sigma**2
    N = 2 * b / sigma**2
    K_disc = 1 - np.exp(-r * T)
    return (-(N - 1) - np.sqrt((N - 1) ** 2 + 4 * M / K_disc)) / 2


def _critical_price_call(K: float, T: float, r: float, b: float, sigma: float) -> float:
    Q2 = _q2(K, T, r, b, sigma)

    S = K * max(1.1, np.exp(sigma * np.sqrt(T)))
    for _ in range(_MAX_ITER):
        d1 = _d1(S, K, T, b, sigma)
        E = np.exp((b - r) * T) * norm.cdf(d1)
        c = bs_price(S, K, T, r, r - b, sigma, "call")
        g = (S - K) - c - (1 - E) * S / Q2
        g_prime = (1 - E) * (1 - 1 / Q2) + np.exp((b - r) * T) * norm.pdf(d1) / (Q2 * sigma * np.sqrt(T))
        step = g / g_prime
        S_new = S - step
        if S_new <= 0:
            S_new = S / 2
        if abs(S_new - S) < _TOL * K:
            return S_new
        S = S_new
    return S


def _critical_price_put(K: float, T: float, r: float, b: float, sigma: float) -> float:
    Q1 = _q1(K, T, r, b, sigma)

    S = K * min(0.9, np.exp(-sigma * np.sqrt(T)))
    for _ in range(_MAX_ITER):
        d1 = _d1(S, K, T, b, sigma)
        E1 = np.exp((b - r) * T) * norm.cdf(-d1)
        p = bs_price(S, K, T, r, r - b, sigma, "put")
        h = (K - S) - p + (1 - E1) * S / Q1
        h_prime = -(1 - E1) * (1 - 1 / Q1) + np.exp((b - r) * T) * norm.pdf(d1) / (Q1 * sigma * np.sqrt(T))
        step = h / h_prime
        S_new = S - step
        if S_new <= 0:
            S_new = S / 2
        if abs(S_new - S) < _TOL * K:
            return S_new
        S = S_new
    return S


def baw_price(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str) -> float:
    """American option price via the Barone-Adesi-Whaley quadratic approximation."""
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    b = r - q

    if option_type == "call":
        if b >= r:
            # No dividend drag on the forward: early exercise is never optimal.
            return bs_price(S, K, T, r, q, sigma, "call")
        try:
            S_star = _critical_price_call(K, T, r, b, sigma)
            Q2 = _q2(K, T, r, b, sigma)
            d1_star = _d1(S_star, K, T, b, sigma)
            A2 = (S_star / Q2) * (1 - np.exp((b - r) * T) * norm.cdf(d1_star))
            if S < S_star:
                return bs_price(S, K, T, r, q, sigma, "call") + A2 * (S / S_star) ** Q2
            return S - K
        except (ValueError, FloatingPointError, ZeroDivisionError):
            return bs_price(S, K, T, r, q, sigma, "call")

    try:
        S_star = _critical_price_put(K, T, r, b, sigma)
        Q1 = _q1(K, T, r, b, sigma)
        d1_star = _d1(S_star, K, T, b, sigma)
        A1 = -(S_star / Q1) * (1 - np.exp((b - r) * T) * norm.cdf(-d1_star))
        if S > S_star:
            return bs_price(S, K, T, r, q, sigma, "put") + A1 * (S / S_star) ** Q1
        return K - S
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return bs_price(S, K, T, r, q, sigma, "put")
