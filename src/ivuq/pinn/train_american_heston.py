"""Training loop for N2-Heston: the coupled solution+boundary network for
the American put under Heston (`heston_free_boundary.py`). Eight loss terms
each epoch:

  1. PDE residual (`heston_pde.pde_residual`, equation (4)), on interior
     points restricted to the current continuation-region estimate
     (`heston_free_boundary.sample_continuation_interior`).
  2. Obstacle floor (`heston_free_boundary.obstacle_residual`): u >= h(m)
     on those same continuation-region points -- true by definition of the
     region, and needed in practice, since nothing else stops the soft PDE
     fit from sagging below intrinsic value while it's still converging.
  3. Value matching (5), at the boundary network's own estimate.
  4. Smooth pasting (6), same points.
  5. Terminal condition on the solution net: u(m, v, 0) = h(m).
  6. Terminal condition on the boundary net: b(v, 0) = 1 (at expiry the
     exercise boundary collapses onto the strike -- classical result).
  7. Far-field m -> m_max (reused from `black_scholes_pde.boundary_high`,
     independent of v -- same as the European/LCP cases).
  8. Far-field v -> v_max (reused from `heston_pde.boundary_v_max_residual`),
     evaluated in the continuation region at that v slice.

No weighting-scheme ablation here (that was N2/GBM's own Phase 3 study,
Section 5/6 of the roadmap) -- fixed weights throughout, same as N1-Heston.
Both networks share one optimizer so their coupling (see
`heston_free_boundary.py`) flows through gradient descent naturally, not
via any alternating-update scheme.
"""

from __future__ import annotations

import torch

from .black_scholes_pde import boundary_high, terminal_condition
from .config import AmericanHestonPINNConfig
from .heston_free_boundary import (
    BoundaryNet,
    boundary_residuals,
    obstacle_residual,
    sample_continuation_interior,
    sample_v_tau,
)
from .heston_pde import boundary_v_max_residual, pde_residual
from .network import PINN

__all__ = ["train_american_heston_pinn"]


def train_american_heston_pinn(
    config: AmericanHestonPINNConfig,
) -> tuple[PINN, BoundaryNet, dict[str, list[float]]]:
    torch.manual_seed(config.seed)
    sol_net = PINN(config.sol_hidden_layers, config.sol_hidden_width, in_dim=3)
    bnd_net = BoundaryNet(config.bnd_hidden_layers, config.bnd_hidden_width)

    optimizer = torch.optim.Adam(list(sol_net.parameters()) + list(bnd_net.parameters()), lr=config.lr)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=0.5)
        if config.lr_decay_every > 0
        else None
    )

    history: dict[str, list[float]] = {
        "total": [],
        "pde": [],
        "obstacle": [],
        "value_matching": [],
        "smooth_pasting": [],
        "terminal_u": [],
        "terminal_b": [],
        "boundary_m_max": [],
        "boundary_v_max": [],
    }

    for _ in range(config.epochs):
        optimizer.zero_grad()

        m_int, v_int, tau_int = sample_continuation_interior(
            bnd_net, config.n_interior, config.m_max, config.v_max, config.tau_max
        )
        pde_r = pde_residual(
            sol_net, m_int, v_int, tau_int,
            config.r, config.q, config.kappa, config.theta, config.xi, config.rho,
        )
        loss_pde = torch.mean(pde_r**2)
        loss_obstacle = torch.mean(obstacle_residual(sol_net, m_int, v_int, tau_int) ** 2)

        v_bc, tau_bc = sample_v_tau(config.n_boundary_vt, config.v_max, config.tau_max)
        value_matching, smooth_pasting, _ = boundary_residuals(sol_net, bnd_net, v_bc, tau_bc)
        loss_value_matching = torch.mean(value_matching**2)
        loss_smooth_pasting = torch.mean(smooth_pasting**2)

        m_term = torch.rand(config.n_terminal, 1) * config.m_max
        v_term = torch.rand(config.n_terminal, 1) * config.v_max
        tau_term = torch.zeros(config.n_terminal, 1)
        u_term_pred = sol_net(torch.cat([m_term, v_term, tau_term], dim=1))
        u_term_true = terminal_condition(m_term, "put")
        loss_terminal_u = torch.mean((u_term_pred - u_term_true) ** 2)

        v_term_b = torch.rand(config.n_boundary_vt, 1) * config.v_max
        tau_term_b = torch.zeros(config.n_boundary_vt, 1)
        b_term_pred = bnd_net(v_term_b, tau_term_b)
        loss_terminal_b = torch.mean((b_term_pred - 1.0) ** 2)

        tau_mmax = torch.rand(config.n_boundary_m_max, 1) * config.tau_max
        v_mmax = torch.rand(config.n_boundary_m_max, 1) * config.v_max
        m_mmax = torch.full_like(tau_mmax, config.m_max)
        u_mmax_pred = sol_net(torch.cat([m_mmax, v_mmax, tau_mmax], dim=1))
        u_mmax_true = boundary_high(tau_mmax, config.m_max, config.r, config.q, "put")
        loss_boundary_m_max = torch.mean((u_mmax_pred - u_mmax_true) ** 2)

        tau_vmax = torch.rand(config.n_boundary_v_max, 1) * config.tau_max
        with torch.no_grad():
            b_vmax = bnd_net(torch.full_like(tau_vmax, config.v_max), tau_vmax)
        m_vmax = b_vmax + torch.rand(config.n_boundary_v_max, 1) * (config.m_max - b_vmax)
        vmax_res = boundary_v_max_residual(sol_net, m_vmax, tau_vmax, config.v_max)
        loss_boundary_v_max = torch.mean(vmax_res**2)

        loss = (
            config.lambda_pde * loss_pde
            + config.lambda_obstacle * loss_obstacle
            + config.lambda_value_matching * loss_value_matching
            + config.lambda_smooth_pasting * loss_smooth_pasting
            + config.lambda_terminal_u * loss_terminal_u
            + config.lambda_terminal_b * loss_terminal_b
            + config.lambda_boundary_m_max * loss_boundary_m_max
            + config.lambda_boundary_v_max * loss_boundary_v_max
        )
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        history["total"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["obstacle"].append(loss_obstacle.item())
        history["value_matching"].append(loss_value_matching.item())
        history["smooth_pasting"].append(loss_smooth_pasting.item())
        history["terminal_u"].append(loss_terminal_u.item())
        history["terminal_b"].append(loss_terminal_b.item())
        history["boundary_m_max"].append(loss_boundary_m_max.item())
        history["boundary_v_max"].append(loss_boundary_v_max.item())

    return sol_net, bnd_net, history
