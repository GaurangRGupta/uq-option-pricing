"""The Heston PDE, non-dimensionalized the same way as Black-Scholes
(`black_scholes_pde.py`), and its residual/boundary terms.

Derivation. The Heston model adds a second state variable, the
instantaneous variance v, as its own square-root diffusion:

    dS = (r-q) S dt + sqrt(v) S dW1
    dv = kappa(theta - v) dt + xi sqrt(v) dW2,     dW1 dW2 = rho dt

The corresponding PDE for a price V(S, v, t), by the usual Feynman-Kac /
hedging argument (Heston, 1993), is:

    dV/dt + 0.5 v S^2 d2V/dS2 + rho xi v S d2V/dSdv + 0.5 xi^2 v d2V/dv2
          + (r-q) S dV/dS + kappa(theta-v) dV/dv - r V = 0

Same non-dimensionalization as the BS case: m = S/K, u(m, v, tau) =
V(K*m, v, tau) / K, tau = T - t. v is already dimensionless (it's a
variance), so it's left alone. Substituting V = K*u(S/K, v, tau) and
dividing through by K gives:

    du/dtau = 0.5 v m^2 d2u/dm2 + rho xi v m d2u/dmdv + 0.5 xi^2 v d2u/dv2
            + (r-q) m du/dm + kappa(theta-v) du/dv - r u                 (4)

This is what `pde_residual` below evaluates via autograd, one line per
term, same style as equation (2) in `black_scholes_pde.py`. Setting xi=0
and holding v fixed at sigma^2 collapses (4) exactly onto (2) -- variance
stops moving, so the two extra terms (the d2u/dv2 diffusion and the
mixed d2u/dmdv term) both vanish and kappa(theta-v)=0 removes the drift-in-v
term too. `tests/test_pinn_european_heston.py` checks this limit directly
against `black_scholes_pde`'s own residual, not just against a price.

Domain: m in [0, m_max], v in [0, v_max], tau in [0, tau_max].

Terminal condition (tau=0): same payoff as the BS case, u(m, v, 0) = h(m),
independent of v -- reused directly from `black_scholes_pde.terminal_condition`,
not redefined here.

Boundary conditions:
  - m -> 0, m -> m_max: same economic reasoning as BS (S=0 is absorbing,
    deep-ITM/OTM forward value dominates), independent of v -- reused
    directly from `black_scholes_pde.boundary_low`/`boundary_high`.
  - v -> 0: the PDE degenerates on its own here (every v-weighted term in
    (4) vanishes except kappa*theta*du/dv), which is itself a valid
    equation the solution must satisfy -- no separate Dirichlet value is
    needed or well-posed at v=0 (this is the standard treatment in Heston
    finite-difference solvers too; v=0 is a natural boundary, not a
    specified one). We simply keep sampling `pde_residual` at v=0
    collocation points rather than adding a fourth boundary function.
  - v -> v_max: no exact condition exists (the domain is a truncation of
    v in [0, infinity)), so we use the standard artificial boundary
    condition from Heston finite-difference literature (e.g. in't Hout &
    Foulon, 2010): d2u/dv2 = 0 at v=v_max, i.e. u is linear in v out at the
    edge of the truncated domain. `boundary_v_max_residual` evaluates this
    via autograd, the same way `pde_residual` evaluates (4).
"""

from __future__ import annotations

import torch

__all__ = ["pde_residual", "boundary_v_max_residual"]


def pde_residual(
    model: torch.nn.Module,
    m: torch.Tensor,
    v: torch.Tensor,
    tau: torch.Tensor,
    r: float,
    q: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
) -> torch.Tensor:
    """Residual of equation (4) above, evaluated at (m, v, tau) via autograd.

    `m`, `v`, `tau` are (N, 1) tensors. A perfectly-solved PDE gives residual
    0 everywhere; the PDE loss term is the mean square of this.
    """
    m = m.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    tau = tau.clone().requires_grad_(True)

    u = model(torch.cat([m, v, tau], dim=1))
    ones = torch.ones_like(u)

    du_dm, du_dv, du_dtau = torch.autograd.grad(
        u, (m, v, tau), grad_outputs=ones, create_graph=True
    )
    d2u_dm2 = torch.autograd.grad(du_dm, m, grad_outputs=torch.ones_like(du_dm), create_graph=True)[0]
    d2u_dv2 = torch.autograd.grad(du_dv, v, grad_outputs=torch.ones_like(du_dv), create_graph=True)[0]
    d2u_dmdv = torch.autograd.grad(du_dm, v, grad_outputs=torch.ones_like(du_dm), create_graph=True)[0]

    return (
        du_dtau
        - 0.5 * v * m**2 * d2u_dm2
        - rho * xi * v * m * d2u_dmdv
        - 0.5 * xi**2 * v * d2u_dv2
        - (r - q) * m * du_dm
        - kappa * (theta - v) * du_dv
        + r * u
    )


def boundary_v_max_residual(
    model: torch.nn.Module,
    m: torch.Tensor,
    tau: torch.Tensor,
    v_max: float,
) -> torch.Tensor:
    """d2u/dv2 at v=v_max -- the artificial far-variance boundary condition.
    `m` and `tau` are (N, 1) tensors; v is fixed at v_max for all of them."""
    m = m.clone().requires_grad_(True)
    v = torch.full_like(m, v_max, requires_grad=True)

    u = model(torch.cat([m, v, tau], dim=1))
    du_dv = torch.autograd.grad(u, v, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dv2 = torch.autograd.grad(du_dv, v, grad_outputs=torch.ones_like(du_dv), create_graph=True)[0]
    return d2u_dv2
