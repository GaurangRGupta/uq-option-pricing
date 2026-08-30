"""The American option as a linear complementarity problem (LCP), and its
soft-penalty loss terms.

Let L[u] be the exact same Black-Scholes operator as the European case
(equation (2) in `black_scholes_pde.py`), evaluated the same way via
autograd:

    L[u] = du/dtau - (r-q) m du/dm - 0.5 sigma^2 m^2 d2u/dm2 + r u

For a European option, L[u] = 0 everywhere (that's the whole PDE). An
American holder can always fall back on immediate exercise, worth the
intrinsic value h(m) = max(m-1, 0) for a call or max(1-m, 0) for a put, so
u >= h(m) everywhere -- the option can never be worth less than exercising
now. Where u > h(m) (the continuation region), holding is strictly better
than exercising, so the option must be priced exactly like a European one
there: L[u] = 0. Where u = h(m) (the exercise region), the holder is
indifferent-or-worse about waiting, which requires L[h] >= 0 there (if
L[h] were negative, waiting would be strictly better and u = h(m) couldn't
be the true value). Together this is the linear complementarity problem:

    L[u] >= 0,      u >= h(m),      L[u] * (u - h(m)) = 0            (3)

The third condition is complementary slackness: at every (m, tau), at least
one of the two inequalities must bind exactly (either the PDE holds with
equality, or the option is worth exactly its intrinsic value). The
unknown curve separating the two regions is the early-exercise boundary --
we never locate it explicitly; (3) enforces it implicitly everywhere.

Terminal condition u(m, 0) = h(m) and the two spatial boundary conditions
are unchanged from the European case (at expiry, deep out-of-the-money, or
deep in-the-money, American and European have the same value) -- reused
directly from `black_scholes_pde.py`, not re-derived here.

As a soft-penalty loss (the standard PINN treatment of a variational
inequality -- see Dhiman & Hu, arXiv:2312.06711, and the coupled-PINN Heston
paper in `planning/papers/PAPER_TRAIL.md`, both of which penalize (3) this
way rather than solving it exactly), each of the three conditions in (3)
becomes a term that is zero exactly when that condition holds:

    ineq_violation            = relu(-L[u])                # >0 iff L[u] < 0
    obstacle_violation        = relu(h(m) - u)              # >0 iff u < h(m)
    complementarity_violation = L[u] * (u - h(m))           # should be ~0

`free_boundary_residuals` returns these three (unreduced) tensors; callers
square and weight them into the training loss.
"""

from __future__ import annotations

import torch

from .black_scholes_pde import pde_residual, terminal_condition as intrinsic_value

__all__ = ["intrinsic_value", "free_boundary_residuals"]


def free_boundary_residuals(
    model: torch.nn.Module,
    m: torch.Tensor,
    tau: torch.Tensor,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three (unreduced, per-point) terms of the LCP penalty (3) above.

    Returns (ineq_violation, obstacle_violation, complementarity_violation),
    each an (N, 1) tensor. `m` and `tau` are the interior collocation points;
    `pde_residual` already runs `model` once via autograd to get L[u], so we
    call it here rather than duplicating the derivative machinery.
    """
    l_u = pde_residual(model, m, tau, r, q, sigma)
    u = model(torch.cat([m, tau], dim=1))
    h = intrinsic_value(m, option_type)

    ineq_violation = torch.relu(-l_u)
    obstacle_violation = torch.relu(h - u)
    complementarity_violation = l_u * (u - h)

    return ineq_violation, obstacle_violation, complementarity_violation
