"""Cox-Ross-Rubinstein binomial tree: European and American option pricing.

This is the American workhorse pricer (BUILD_PLAN.md Part 2). It is written
in plain vectorized numpy for correctness first. The numba @njit speed-up
(BUILD_PLAN.md 6.3, needed once we run this across a full dataset instead of
one-off calls) is a follow-up: swap the backward-induction loop body for a
@njit-compiled version behind the same crr_price() signature, nothing else
in the codebase needs to change.
"""

from __future__ import annotations

import numpy as np

__all__ = ["crr_price"]


def crr_price(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
    is_american: bool,
    steps: int = 500,
) -> float:
    """Price a European or American option with an N-step CRR binomial tree."""
    option_type = option_type.lower()
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if T <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    growth = np.exp((r - q) * dt)
    p = (growth - d) / (u - d)
    if not (0.0 < p < 1.0):
        raise ValueError(
            f"risk-neutral probability p={p:.4f} is outside (0, 1); "
            "steps too coarse for this sigma/T combination"
        )
    discount = np.exp(-r * dt)

    j = np.arange(steps + 1)
    spot_at_expiry = S * u**j * d ** (steps - j)
    if option_type == "call":
        values = np.maximum(spot_at_expiry - K, 0.0)
    else:
        values = np.maximum(K - spot_at_expiry, 0.0)

    for step in range(steps - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        if is_american:
            j = np.arange(step + 1)
            spot = S * u**j * d ** (step - j)
            exercise = np.maximum(spot - K, 0.0) if option_type == "call" else np.maximum(K - spot, 0.0)
            values = np.maximum(values, exercise)

    return float(values[0])
