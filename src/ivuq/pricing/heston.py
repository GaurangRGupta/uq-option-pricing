"""Heston (1993) European option price via characteristic-function inversion.

The Heston model:

    dS = (r-q) S dt + sqrt(v) S dW1
    dv = kappa(theta - v) dt + xi sqrt(v) dW2,      dW1 dW2 = rho dt

has a semi-closed-form European price via Fourier inversion of its
characteristic function:

    C = S exp(-q T) P1 - K exp(-r T) P2
    P_j = 1/2 + (1/pi) integral_0^inf Re[ exp(-i*phi*ln K) f_j(phi) / (i*phi) ] dphi

Implemented here using the "Little Trap" sign convention (Albrecher, Mayer,
Schoutens & Tistaert, 2007, "The Little Heston Trap", Wilmott Magazine): the
original 1993 formula for f_j has a branch-cut in the complex logarithm that
makes the naive formula numerically unstable at long maturities / certain
parameter combinations, and the Little Trap is the standard fix used by
every production implementation, applied to the same 1993 closed form
(nothing about the model changes, only which root of a quadratic gets
picked). This module is unrelated to any of the 4 papers in
`planning/papers/PAPER_TRAIL.md` -- none of them touch Heston's own
closed-form solution -- it exists purely as our own independent reference
price, the same role `black_scholes.py` plays for N1 and `binomial.py`
plays for N2. American options under Heston have no closed form at all,
which is exactly why N2-Heston will need its own validation instrument
(a Longstaff-Schwartz Monte Carlo reference, not built yet).

Put price via put-call parity: P = C - S exp(-q T) + K exp(-r T).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

__all__ = ["heston_price"]

_PHI_MIN = 1e-10
_PHI_MAX = 200.0


def _characteristic_integrand(phi: float, S: float, K: float, T: float, r: float, q: float,
                               kappa: float, theta: float, xi: float, rho: float, v0: float, j: int) -> float:
    i = 1j
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho * xi if j == 1 else kappa
    a = kappa * theta

    d = np.sqrt((rho * xi * i * phi - b) ** 2 - xi**2 * (2 * u * i * phi - phi**2))
    g = (b - rho * xi * i * phi - d) / (b - rho * xi * i * phi + d)
    exp_dT = np.exp(-d * T)

    C = i * phi * (r - q) * T + (a / xi**2) * (
        (b - rho * xi * i * phi - d) * T - 2 * np.log((1 - g * exp_dT) / (1 - g))
    )
    D = ((b - rho * xi * i * phi - d) / xi**2) * ((1 - exp_dT) / (1 - g * exp_dT))
    f = np.exp(C + D * v0 + i * phi * np.log(S))

    return float(np.real(np.exp(-i * phi * np.log(K)) * f / (i * phi)))


def _probability(S: float, K: float, T: float, r: float, q: float,
                  kappa: float, theta: float, xi: float, rho: float, v0: float, j: int) -> float:
    integral, _ = quad(
        _characteristic_integrand, _PHI_MIN, _PHI_MAX,
        args=(S, K, T, r, q, kappa, theta, xi, rho, v0, j), limit=200,
    )
    return 0.5 + integral / np.pi


def heston_price(
    S: float, K: float, T: float, r: float, q: float,
    kappa: float, theta: float, xi: float, rho: float, v0: float,
    option_type: str,
) -> float:
    """European option price under Heston dynamics.

    kappa: mean-reversion speed of variance. theta: long-run variance.
    xi: vol-of-vol. rho: correlation between the two Brownian motions.
    v0: initial variance (the state variable, not a volatility -- pass
    sigma**2 if you want to compare against a constant-vol GBM price).
    """
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    P1 = _probability(S, K, T, r, q, kappa, theta, xi, rho, v0, j=1)
    P2 = _probability(S, K, T, r, q, kappa, theta, xi, rho, v0, j=2)
    call = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2

    if option_type == "call":
        return call
    return call - S * np.exp(-q * T) + K * np.exp(-r * T)
