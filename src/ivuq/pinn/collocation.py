"""Random collocation-point sampling for the three loss terms.

Uniform random sampling, resampled fresh every training epoch, so the
network sees the whole domain over training rather than memorizing a fixed
grid -- standard PINN practice, and cheap since sampling is O(n).
"""

from __future__ import annotations

import torch

from .free_boundary import intrinsic_value

__all__ = ["sample_interior", "sample_terminal", "sample_boundary_tau", "sample_interior_near_boundary"]


def sample_interior(n: int, m_max: float, tau_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Points in the open domain (0, m_max) x (0, tau_max], for the PDE loss."""
    m = torch.rand(n, 1) * m_max
    tau = torch.rand(n, 1) * tau_max
    return m, tau


def sample_terminal(n: int, m_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Points on the tau=0 slice, for the terminal (payoff) loss.

    Half uniform over the whole domain, half concentrated near m=1 (a
    Gaussian clipped to stay in range). The payoff has a kink at m=1; a
    smooth network needs denser samples there to resolve it, the same way
    you'd refine a numerical grid near a non-smooth point.
    """
    n_uniform = n // 2
    n_kink = n - n_uniform
    m_uniform = torch.rand(n_uniform, 1) * m_max
    m_kink = (1.0 + 0.15 * torch.randn(n_kink, 1)).clamp(0.0, m_max)
    m = torch.cat([m_uniform, m_kink], dim=0)
    tau = torch.zeros(n, 1)
    return m, tau


def sample_boundary_tau(n: int, tau_max: float) -> torch.Tensor:
    """tau values to pair with m=0 and m=m_max, for the two boundary losses."""
    return torch.rand(n, 1) * tau_max


def sample_interior_near_boundary(
    model: torch.nn.Module,
    n: int,
    m_max: float,
    tau_max: float,
    option_type: str,
    band: float = 0.1,
    oversample: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interior points concentrated near the model's *current* estimate of the
    early-exercise boundary -- the curriculum weighting scheme's adaptive
    resampling (see `train_american.py`).

    The exercise boundary is exactly where u(m, tau) is close to the
    intrinsic value h(m); we don't know it in closed form, so we draw a
    large uniform candidate pool, keep the ones where the model currently
    thinks it's close to that boundary (rejection sampling), and top up with
    plain uniform points if too few candidates qualify (early in training,
    before the model has any idea where the boundary is, this is the common
    case and just falls back to uniform sampling)."""
    with torch.no_grad():
        m_pool = torch.rand(n * oversample, 1) * m_max
        tau_pool = torch.rand(n * oversample, 1) * tau_max
        u_pool = model(torch.cat([m_pool, tau_pool], dim=1))
        h_pool = intrinsic_value(m_pool, option_type)
        close = (u_pool - h_pool).abs() < band

    m_near = m_pool[close.squeeze(-1)][:n]
    tau_near = tau_pool[close.squeeze(-1)][:n]

    n_fill = n - m_near.shape[0]
    if n_fill > 0:
        m_fill = torch.rand(n_fill, 1) * m_max
        tau_fill = torch.rand(n_fill, 1) * tau_max
        m_near = torch.cat([m_near, m_fill], dim=0)
        tau_near = torch.cat([tau_near, tau_fill], dim=0)

    return m_near, tau_near
