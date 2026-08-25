"""The Black-Scholes PDE, non-dimensionalized, and its residual/boundary terms.

Derivation (so the code below is checkable against it):

Start from the standard Black-Scholes-Merton PDE for a price V(S, t) with
strike K, time-to-expiry tau = T - t, rate r, dividend yield q, vol sigma:

    dV/dtau = (r-q) S dV/dS + 0.5 sigma^2 S^2 d2V/dS2 - r V          (1)

with terminal condition at tau=0 (expiry): V(S, 0) = payoff(S).

Non-dimensionalize by moneyness m = S/K and u(m, tau) = V(K*m, tau) / K
(so u is the price in units of strike). Substituting V = K*u(S/K, tau) into
(1) and dividing through by K gives:

    du/dtau = (r-q) m du/dm + 0.5 sigma^2 m^2 d2u/dm2 - r u          (2)

This is what `pde_residual` below evaluates directly via autograd -- one line
per term of (2), nothing hidden. Working in (m, tau) instead of (S, t) keeps
K out of the network's inputs entirely: one trained model prices any strike
at that (r, q, sigma), you just rescale by K.

Terminal condition (tau=0): u(m, 0) = max(m-1, 0) for a call, max(1-m, 0)
for a put -- the payoff in strike units.

Boundary conditions come from the economics at the domain edges, not from
the PDE:
  - m -> 0 (S=0 is absorbing under GBM, so S stays at 0 to expiry):
      call is worthless:          u(0, tau) = 0
      put pays K for certain:     u(0, tau) = exp(-r*tau)
  - m -> m_max (deep in the money, forward value dominates):
      call:  u(m_max, tau) = m_max * exp(-q*tau) - exp(-r*tau)
      put:   u(m_max, tau) = 0
"""

from __future__ import annotations

import torch

__all__ = ["pde_residual", "terminal_condition", "boundary_low", "boundary_high"]


def pde_residual(
    model: torch.nn.Module,
    m: torch.Tensor,
    tau: torch.Tensor,
    r: float,
    q: float,
    sigma: float,
) -> torch.Tensor:
    """Residual of equation (2) above, evaluated at (m, tau) via autograd.

    `m` and `tau` are (N, 1) tensors. A perfectly-solved PDE gives residual 0
    everywhere; the PDE loss term is the mean square of this.
    """
    m = m.clone().requires_grad_(True)
    tau = tau.clone().requires_grad_(True)

    u = model(torch.cat([m, tau], dim=1))
    ones = torch.ones_like(u)

    du_dm, du_dtau = torch.autograd.grad(u, (m, tau), grad_outputs=ones, create_graph=True)
    d2u_dm2 = torch.autograd.grad(du_dm, m, grad_outputs=torch.ones_like(du_dm), create_graph=True)[0]

    return du_dtau - (r - q) * m * du_dm - 0.5 * sigma**2 * m**2 * d2u_dm2 + r * u


def terminal_condition(m: torch.Tensor, option_type: str) -> torch.Tensor:
    """Payoff in strike units at tau=0."""
    if option_type == "call":
        return torch.clamp(m - 1.0, min=0.0)
    return torch.clamp(1.0 - m, min=0.0)


def boundary_low(tau: torch.Tensor, r: float, option_type: str) -> torch.Tensor:
    """u(m=0, tau)."""
    if option_type == "call":
        return torch.zeros_like(tau)
    return torch.exp(-r * tau)


def boundary_high(tau: torch.Tensor, m_max: float, r: float, q: float, option_type: str) -> torch.Tensor:
    """u(m=m_max, tau)."""
    if option_type == "call":
        return m_max * torch.exp(-q * tau) - torch.exp(-r * tau)
    return torch.zeros_like(tau)
