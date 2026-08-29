"""Training loop: minimize the weighted sum of the PDE, terminal, and
boundary losses (equation (2) and its terminal/boundary conditions in
`black_scholes_pde.py`) with Adam, plus a simple step-decay on the learning
rate (PINN loss landscapes are typically fine near a minimum but noisy
getting there, so a smaller late-training step size helps convergence).
"""

from __future__ import annotations

import torch

from .black_scholes_pde import boundary_high, boundary_low, pde_residual, terminal_condition
from .collocation import sample_boundary_tau, sample_interior, sample_terminal
from .config import EuropeanPINNConfig
from .network import PINN

__all__ = ["train_european_pinn"]


def train_european_pinn(config: EuropeanPINNConfig) -> tuple[PINN, dict[str, list[float]]]:
    torch.manual_seed(config.seed)
    model = PINN(config.hidden_layers, config.hidden_width)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=0.5)
        if config.lr_decay_every > 0
        else None
    )

    history: dict[str, list[float]] = {"total": [], "pde": [], "terminal": [], "boundary": []}

    for _ in range(config.epochs):
        optimizer.zero_grad()

        m_int, tau_int = sample_interior(config.n_interior, config.m_max, config.tau_max)
        residual = pde_residual(model, m_int, tau_int, config.r, config.q, config.sigma)
        loss_pde = torch.mean(residual**2)

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

        loss = (
            config.lambda_pde * loss_pde
            + config.lambda_terminal * loss_terminal
            + config.lambda_boundary * loss_boundary
        )
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        history["total"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["terminal"].append(loss_terminal.item())
        history["boundary"].append(loss_boundary.item())

    return model, history
