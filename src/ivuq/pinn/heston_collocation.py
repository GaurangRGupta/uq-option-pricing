"""Random collocation-point sampling for the Heston PDE's loss terms --
same "uniform, resampled fresh every epoch" convention as `collocation.py`,
just over the extra v dimension.
"""

from __future__ import annotations

import torch

__all__ = [
    "sample_interior",
    "sample_terminal",
    "sample_boundary_m",
    "sample_boundary_v_max",
]


def sample_interior(
    n: int, m_max: float, v_max: float, tau_max: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Points in the open domain (0,m_max) x (0,v_max) x (0,tau_max], for the
    Heston PDE loss."""
    m = torch.rand(n, 1) * m_max
    v = torch.rand(n, 1) * v_max
    tau = torch.rand(n, 1) * tau_max
    return m, v, tau


def sample_terminal(n: int, m_max: float, v_max: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Points on the tau=0 slice, for the terminal (payoff) loss. Same
    kink-concentration in m as the GBM case (`collocation.sample_terminal`)
    -- the payoff's kink at m=1 is still there regardless of v -- with v
    sampled uniformly since the payoff doesn't depend on it."""
    n_uniform = n // 2
    n_kink = n - n_uniform
    m_uniform = torch.rand(n_uniform, 1) * m_max
    m_kink = (1.0 + 0.15 * torch.randn(n_kink, 1)).clamp(0.0, m_max)
    m = torch.cat([m_uniform, m_kink], dim=0)
    v = torch.rand(n, 1) * v_max
    tau = torch.zeros(n, 1)
    return m, v, tau


def sample_boundary_m(n: int, v_max: float, tau_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(v, tau) pairs to pair with m=0 and m=m_max, for the two m-boundary
    losses (reused from `black_scholes_pde.boundary_low`/`boundary_high`,
    which don't depend on v)."""
    v = torch.rand(n, 1) * v_max
    tau = torch.rand(n, 1) * tau_max
    return v, tau


def sample_boundary_v_max(n: int, m_max: float, tau_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(m, tau) pairs to evaluate the artificial v=v_max boundary condition
    (`heston_pde.boundary_v_max_residual`) at."""
    m = torch.rand(n, 1) * m_max
    tau = torch.rand(n, 1) * tau_max
    return m, tau
