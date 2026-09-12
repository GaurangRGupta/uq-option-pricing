"""The American free-boundary problem under Heston, formulated as a coupled
solution network + boundary network with classical value-matching and
smooth-pasting conditions -- not the LCP/obstacle-penalty treatment
`free_boundary.py` uses for GBM (N2). Two different, independently valid
ways to handle the same variational inequality; the 2026-08-30 DEVLOG entry
has the reasoning for why this project uses the LCP route for GBM and the
dual-network route for Heston (the latter is what Phase 5's exercise-
boundary conformal band will need an explicit boundary function for, which
the LCP's single network never produces).

Setup. Under GBM the free boundary is a curve b(tau) in one dimension: below
it (for a put) the option is worth exactly its intrinsic value, above it the
Black-Scholes PDE holds. Under Heston, variance is a second state variable,
so the free boundary is a *surface* b(v, tau) -- for every (v, tau) there is
a moneyness level below which early exercise is optimal.

Classical free-boundary theory (McKean, 1965; see also Wilmott, Howison &
Dewynne, "The Mathematics of Financial Derivatives", ch. 8) gives two
conditions that pin the boundary down, on top of the PDE holding in the
continuation region and the option equalling intrinsic value in the
exercise region:

    u(b(v,tau), v, tau) = h(b(v,tau))                                 (5)
    du/dm (b(v,tau), v, tau) = h'(b(v,tau))                           (6)

(5) is value matching: the price is continuous across the boundary, no
arbitrage jump. (6) is smooth pasting: the price meets intrinsic value
tangentially, not at a kink -- this is the extra condition, beyond mere
continuity, that singles out the *correct* boundary among every other
possible early-exercise policy (any other choice of boundary can still be
made value-matching by construction, but only the true optimal boundary is
also smooth-pasting there).

For a put, h(m) = max(1-m, 0). The boundary always sits below the strike
(b < 1 -- a classical fact: an American put's early-exercise region can
never extend to where intrinsic value would be non-positive), so h(b) = 1-b
and h'(b) = -1 exactly, no kink-crossing ambiguity. This project scopes
N2-Heston to puts only: an American call with q=0 (the default used
throughout this project's own test suite) is never exercised early at all,
which sends the call's boundary to m -> infinity -- unlearnable by a
bounded network, and pointless anyway since American = European call there.
Puts always have a genuine, bounded early-exercise region, which is where
the dual-network machinery actually earns its keep. Paper 3 in
`planning/papers/PAPER_TRAIL.md` makes the identical scoping choice for the
identical reason -- independently arrived at here, not copied from it.

By definition, the continuation region is exactly where u > h(m) (that is
what distinguishes it from the exercise region) -- so any point sampled by
`sample_continuation_interior` should satisfy u >= h(m) once the network is
well-trained, the same obstacle inequality `free_boundary.py` enforces for
the LCP's single network (equation (3) there), just restricted here to the
continuation-only points rather than the whole domain. `obstacle_residual`
penalizes violations of it as an auxiliary loss term: not one of the two
classical free-boundary conditions (5)/(6) above, but a direct consequence
of them that is cheap to check and keeps training from wandering into
economically impossible (negative, or sub-intrinsic) prices while the PDE
fit is still converging elsewhere in the domain.

Two networks, `BoundaryNet` wrapping a plain `network.PINN`:
  - the solution network u_theta(m, v, tau) (in_dim=3, same shape as
    N1-Heston's network) -- meaningful only in the continuation region
    m > b(v, tau). It is never asked to represent the exercise region at
    all, unlike the LCP's single network, which has to cover both regions
    and rely on the obstacle penalty to stay above intrinsic value.
  - the boundary network b_phi(v, tau) (in_dim=2), squashed through a
    sigmoid so its output always lies in (0, 1) -- exploiting the classical
    b < 1 fact above directly, rather than leaving an unconstrained network
    to discover it from data alone.

Training couples the two deliberately: `boundary_residuals` evaluates
(5)/(6) at m = b_phi(v, tau) itself, not at some independently-sampled m --
so gradients from both loss terms flow into *both* networks every step. The
solution network's local slope shapes what boundary matches it; the
boundary's position shapes what the solution network is asked to match at.
`sample_continuation_interior` restricts PDE collocation to
m > b_phi(v, tau).detach() -- a hard domain restriction (the PDE is only
valid in the continuation region to begin with), adaptive to the current
boundary estimate as training progresses, in the same spirit as (but a
stronger constraint than) `collocation.sample_interior_near_boundary`'s
concentration weighting on the GBM/LCP side.
"""

from __future__ import annotations

import torch

from .black_scholes_pde import terminal_condition as intrinsic_value
from .network import PINN

__all__ = [
    "BoundaryNet",
    "boundary_residuals",
    "obstacle_residual",
    "sample_continuation_interior",
    "sample_v_tau",
]


class BoundaryNet(torch.nn.Module):
    """Wraps a plain `PINN(in_dim=2)` with a sigmoid output, so b_phi(v, tau)
    always lies in (0, 1) -- see the module docstring for why that's a
    justified constraint for a put's exercise boundary, not just a
    convenience clamp."""

    def __init__(self, hidden_layers: int, hidden_width: int) -> None:
        super().__init__()
        self.net = PINN(hidden_layers, hidden_width, in_dim=2)

    def forward(self, v: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([v, tau], dim=1)))


def boundary_residuals(
    sol_net: torch.nn.Module,
    bnd_net: BoundaryNet,
    v: torch.Tensor,
    tau: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Value-matching (5) and smooth-pasting (6) residuals, evaluated at the
    boundary network's own current estimate m = b_phi(v, tau) -- couples
    the two networks' gradients together, per the module docstring. Returns
    (value_matching, smooth_pasting, b); `b` is handed back so callers can
    also penalize b(v, tau=0) = 1 (the terminal boundary condition) without
    a second forward pass through `bnd_net`."""
    b = bnd_net(v, tau)
    u = sol_net(torch.cat([b, v, tau], dim=1))
    du_dm = torch.autograd.grad(u, b, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    h = intrinsic_value(b, "put")
    value_matching = u - h
    smooth_pasting = du_dm - (-1.0)

    return value_matching, smooth_pasting, b


def obstacle_residual(sol_net: torch.nn.Module, m: torch.Tensor, v: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """relu(h(m) - u) at continuation-region points -- see the module
    docstring for why this is a justified (not just convenient) auxiliary
    term, restricted to `sample_continuation_interior`'s points."""
    u = sol_net(torch.cat([m, v, tau], dim=1))
    h = intrinsic_value(m, "put")
    return torch.relu(h - u)


def sample_continuation_interior(
    bnd_net: BoundaryNet, n: int, m_max: float, v_max: float, tau_max: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PDE collocation points with m drawn from (b_phi(v, tau), m_max) --
    the continuation region only, per the boundary network's current
    estimate. `bnd_net` is only queried (no_grad) to decide *where* to
    sample; this is not a loss term.

    Price curvature in m is sharpest immediately above the boundary (the
    solution is bending from the payoff's linear intrinsic-value slope
    toward the PDE's own dynamics right where the two regions meet) and
    flattens out toward m_max, so half the points are drawn with a
    quadratic bias toward b (m = b + (m_max-b)*U^2, U~Uniform(0,1), which
    concentrates samples near the low end of the interval) and half
    uniformly, for full-domain coverage -- same "concentrate near the hard
    part of the domain, don't abandon the rest" idea as
    `collocation.sample_interior_near_boundary` on the GBM/LCP side, just a
    sampling-density bias here instead of a rejection filter."""
    v = torch.rand(n, 1) * v_max
    tau = torch.rand(n, 1) * tau_max
    with torch.no_grad():
        b = bnd_net(v, tau)
    n_near = n // 2
    u_near = torch.rand(n_near, 1) ** 2
    u_uniform = torch.rand(n - n_near, 1)
    u = torch.cat([u_near, u_uniform], dim=0)
    m = b + u * (m_max - b)
    return m, v, tau


def sample_v_tau(n: int, v_max: float, tau_max: float) -> tuple[torch.Tensor, torch.Tensor]:
    """(v, tau) pairs for the boundary-network losses: value-matching,
    smooth-pasting, and the terminal condition b(v, 0) = 1."""
    v = torch.rand(n, 1) * v_max
    tau = torch.rand(n, 1) * tau_max
    return v, tau
