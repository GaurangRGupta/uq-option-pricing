"""Labelled training data for the data-driven baselines.

The PINN never sees a labelled price: it is trained from the PDE, the payoff
and the boundary conditions alone. The ML/DL baselines are the opposite, they
learn only from (m, tau) -> u examples, where m = S/K is moneyness and
u = V/K is the price in strike units, exactly the variables the PINN works in
(see `ivuq.pinn.black_scholes_pde`). Using the same variables keeps the
comparison about the learning method, not about input scaling.

Labels come from the classical references already built in this project:

  - European: Black-Scholes closed form.
  - American: the 500-step CRR binomial tree (no closed form exists).

With K = 1 the reference pricers return u directly. Optional additive Gaussian
noise, u_obs = u + eps with eps ~ N(0, noise^2), stands in for the bid-ask
noise a real quote carries; the PINN is unaffected by it because it uses no
labels.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from ivuq.pricing.binomial import crr_price
from ivuq.pricing.black_scholes import price as bs_price

__all__ = ["MarketParams", "Box", "reference_u", "sample_labelled", "add_noise"]


@dataclass(frozen=True)
class MarketParams:
    r: float = 0.03
    q: float = 0.0
    sigma: float = 0.2


@dataclass(frozen=True)
class Box:
    """Axis-aligned region of the (m, tau) plane."""

    m_lo: float
    m_hi: float
    tau_lo: float
    tau_hi: float

    def contains(self, m: np.ndarray, tau: np.ndarray) -> np.ndarray:
        return (m >= self.m_lo) & (m <= self.m_hi) & (tau >= self.tau_lo) & (tau <= self.tau_hi)


def _price_chunk(args: tuple) -> list[float]:
    style, option_type, m, tau, params, steps = args
    out = []
    for mi, ti in zip(m, tau):
        if style == "european":
            out.append(bs_price(mi, 1.0, ti, params.r, params.q, params.sigma, option_type))
        else:
            out.append(crr_price(mi, 1.0, ti, params.r, params.q, params.sigma, option_type, True, steps=steps))
    return out


def reference_u(
    style: str,
    option_type: str,
    m: np.ndarray,
    tau: np.ndarray,
    params: MarketParams,
    tree_steps: int = 500,
    max_workers: int | None = None,
) -> np.ndarray:
    """Reference price in strike units, u(m, tau), for each (m, tau) pair."""
    if style not in ("european", "american"):
        raise ValueError(f"style must be 'european' or 'american', got {style!r}")
    m = np.asarray(m, dtype=np.float64).ravel()
    tau = np.maximum(np.asarray(tau, dtype=np.float64).ravel(), 1e-6)

    workers = max_workers if max_workers is not None else (os.cpu_count() or 1)
    if style == "american" and len(m) >= 200 and workers > 1:
        chunks = np.array_split(np.arange(len(m)), workers)
        jobs = [(style, option_type, m[c], tau[c], params, tree_steps) for c in chunks if len(c)]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_price_chunk, jobs))
        return np.array([x for part in parts for x in part], dtype=np.float64)
    return np.array(_price_chunk((style, option_type, m, tau, params, tree_steps)), dtype=np.float64)


def sample_labelled(
    style: str,
    option_type: str,
    n: int,
    box: Box,
    params: MarketParams,
    seed: int,
    tree_steps: int = 500,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """n uniform draws from `box`, returned as (X, u) with X[:, 0] = m and
    X[:, 1] = tau. Noise-free; add noise with `add_noise` so the same draw can
    be reused across noise levels (the American labels are the expensive part).
    """
    rng = np.random.default_rng(seed)
    m = rng.uniform(box.m_lo, box.m_hi, size=n)
    tau = rng.uniform(box.tau_lo, box.tau_hi, size=n)
    u = reference_u(style, option_type, m, tau, params, tree_steps, max_workers)
    return np.column_stack([m, tau]), u


def add_noise(u: np.ndarray, noise: float, seed: int) -> np.ndarray:
    if noise <= 0.0:
        return u
    rng = np.random.default_rng(seed + 10_000)
    return u + rng.normal(0.0, noise, size=u.shape)
