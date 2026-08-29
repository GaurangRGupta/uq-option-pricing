"""Random collocation-point sampling for the three loss terms.

Uniform random sampling, resampled fresh every training epoch, so the
network sees the whole domain over training rather than memorizing a fixed
grid -- standard PINN practice, and cheap since sampling is O(n).
"""

from __future__ import annotations

import torch

__all__ = ["sample_interior", "sample_terminal", "sample_boundary_tau"]


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
