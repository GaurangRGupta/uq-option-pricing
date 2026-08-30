"""Training loop for the American (N2) free-boundary PINN, with three
loss-weighting schemes selectable via `AmericanPINNConfig.weighting_scheme`
-- Phase 3's ablation (self-adaptive vs. curriculum vs. fixed weighting,
Section 5/6 of the roadmap).

All three share the same network, optimizer, and terminal/boundary losses
(identical in form to the European PINN's, since American and European
agree at expiry and at the domain edges -- see `free_boundary.py`). They
differ only in how the three LCP penalty terms (ineq/obstacle/
complementarity, equation (3) in `free_boundary.py`) get weighted, and
where the interior collocation points come from:

  - "fixed": constant lambda weights, fresh uniform interior sampling every
    epoch -- the simplest possible treatment, and the baseline the other
    two are measured against.
  - "curriculum": the three LCP weights ramp linearly from a small fraction
    of their target value up to the target over `curriculum_ramp_epochs`,
    so the network first settles into roughly the right shape under a weak
    constraint before the free-boundary condition is enforced at full
    strength. Once the ramp completes, interior sampling switches to half
    uniform / half concentrated near the model's current estimate of the
    exercise boundary (`sample_interior_near_boundary`), refining exactly
    where the LCP is hardest to satisfy.
  - "self_adaptive": a simplified, points-only version of McClenny &
    Braga-Neto (arXiv:2009.04544). The interior points are sampled once and
    held fixed for the whole run (self-adaptive weights are tied to
    specific points, so the points can't be resampled out from under them).
    Each of the three LCP terms gets its own per-point trainable weight
    softplus(a_i), updated by *gradient ascent* -- so the network can't
    just learn to ignore a persistently-hard point (e.g. one sitting on the
    free boundary) by driving its own error down while the weight stays
    put; only the weight itself moving up pushes back. Implemented as: run
    the usual backward pass, then flip the sign of the gradient on the
    weight parameters before the (single, shared) optimizer step --
    descending on -loss is the same as ascending on loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .black_scholes_pde import boundary_high, boundary_low, terminal_condition
from .collocation import sample_boundary_tau, sample_interior, sample_interior_near_boundary, sample_terminal
from .config import AmericanPINNConfig
from .free_boundary import free_boundary_residuals
from .network import PINN

__all__ = ["train_american_pinn"]


def _terminal_boundary_losses(
    model: PINN, config: AmericanPINNConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    m_term, tau_term = sample_terminal(config.n_terminal, config.m_max)
    u_term_pred = model(torch.cat([m_term, tau_term], dim=1))
    u_term_true = terminal_condition(m_term, config.option_type)
    loss_terminal = torch.mean((u_term_pred - u_term_true) ** 2)

    tau_bnd = sample_boundary_tau(config.n_boundary, config.tau_max)
    m_low = torch.zeros_like(tau_bnd)
    m_high = torch.full_like(tau_bnd, config.m_max)
    u_low_pred = model(torch.cat([m_low, tau_bnd], dim=1))
    u_high_pred = model(torch.cat([m_high, tau_bnd], dim=1))
    u_low_true = boundary_low(tau_bnd, config.r, config.option_type)
    u_high_true = boundary_high(tau_bnd, config.m_max, config.r, config.q, config.option_type)
    loss_boundary = torch.mean((u_low_pred - u_low_true) ** 2) + torch.mean((u_high_pred - u_high_true) ** 2)
    return loss_terminal, loss_boundary


def train_american_pinn(config: AmericanPINNConfig) -> tuple[PINN, dict[str, list[float]]]:
    torch.manual_seed(config.seed)
    model = PINN(config.hidden_layers, config.hidden_width)

    adaptive_weights: list[torch.nn.Parameter] = []
    m_int_fixed = tau_int_fixed = None
    if config.weighting_scheme == "self_adaptive":
        m_int_fixed, tau_int_fixed = sample_interior(config.n_interior, config.m_max, config.tau_max)
        a_ineq = torch.nn.Parameter(torch.zeros(config.n_interior, 1))
        a_obstacle = torch.nn.Parameter(torch.zeros(config.n_interior, 1))
        a_complementarity = torch.nn.Parameter(torch.zeros(config.n_interior, 1))
        adaptive_weights = [a_ineq, a_obstacle, a_complementarity]

    optimizer = torch.optim.Adam(list(model.parameters()) + adaptive_weights, lr=config.lr)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=0.5)
        if config.lr_decay_every > 0
        else None
    )

    history: dict[str, list[float]] = {
        "total": [],
        "ineq": [],
        "obstacle": [],
        "complementarity": [],
        "terminal": [],
        "boundary": [],
    }

    for epoch in range(config.epochs):
        optimizer.zero_grad()

        fraction = 1.0
        if config.weighting_scheme == "curriculum":
            fraction = min(1.0, epoch / config.curriculum_ramp_epochs) if config.curriculum_ramp_epochs > 0 else 1.0

        if config.weighting_scheme == "self_adaptive":
            m_int, tau_int = m_int_fixed, tau_int_fixed
        elif config.weighting_scheme == "curriculum" and fraction >= 1.0:
            n_near = config.n_interior // 2
            m_near, tau_near = sample_interior_near_boundary(
                model, n_near, config.m_max, config.tau_max, config.option_type
            )
            m_uniform, tau_uniform = sample_interior(config.n_interior - n_near, config.m_max, config.tau_max)
            m_int = torch.cat([m_near, m_uniform], dim=0)
            tau_int = torch.cat([tau_near, tau_uniform], dim=0)
        else:
            m_int, tau_int = sample_interior(config.n_interior, config.m_max, config.tau_max)

        ineq_v, obstacle_v, complementarity_v = free_boundary_residuals(
            model, m_int, tau_int, config.r, config.q, config.sigma, config.option_type
        )

        if config.weighting_scheme == "self_adaptive":
            w_ineq = F.softplus(a_ineq)
            w_obstacle = F.softplus(a_obstacle)
            w_complementarity = F.softplus(a_complementarity)
        elif config.weighting_scheme == "curriculum":
            ramp = config.curriculum_start_frac + (1.0 - config.curriculum_start_frac) * fraction
            w_ineq = config.lambda_ineq * ramp
            w_obstacle = config.lambda_obstacle * ramp
            w_complementarity = config.lambda_complementarity * ramp
        else:
            w_ineq, w_obstacle, w_complementarity = config.lambda_ineq, config.lambda_obstacle, config.lambda_complementarity

        loss_ineq = torch.mean(w_ineq * ineq_v**2)
        loss_obstacle = torch.mean(w_obstacle * obstacle_v**2)
        loss_complementarity = torch.mean(w_complementarity * complementarity_v**2)
        loss_terminal, loss_boundary = _terminal_boundary_losses(model, config)

        loss = (
            loss_ineq
            + loss_obstacle
            + loss_complementarity
            + config.lambda_terminal * loss_terminal
            + config.lambda_boundary * loss_boundary
        )
        loss.backward()

        if config.weighting_scheme == "self_adaptive":
            for p in adaptive_weights:
                if p.grad is not None:
                    p.grad.neg_()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        history["total"].append(loss.item())
        history["ineq"].append(loss_ineq.item())
        history["obstacle"].append(loss_obstacle.item())
        history["complementarity"].append(loss_complementarity.item())
        history["terminal"].append(loss_terminal.item())
        history["boundary"].append(loss_boundary.item())

    return model, history
