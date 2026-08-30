"""Training loop for N0/N1-Heston: minimize the weighted sum of the Heston
PDE loss, the terminal loss, and the three boundary losses (m=0, m=m_max,
v=v_max), same Adam + step-decay recipe as `train.py`'s European/GBM case.
`lambda_pde` is the N0/N1 switch here too (see `config.py`).
"""

from __future__ import annotations

import torch

from .black_scholes_pde import boundary_high, boundary_low, terminal_condition
from .config import HestonEuropeanPINNConfig
from .heston_collocation import sample_boundary_m, sample_boundary_v_max, sample_interior, sample_terminal
from .heston_pde import boundary_v_max_residual, pde_residual
from .network import PINN

__all__ = ["train_european_heston_pinn"]


def train_european_heston_pinn(config: HestonEuropeanPINNConfig) -> tuple[PINN, dict[str, list[float]]]:
    torch.manual_seed(config.seed)
    model = PINN(config.hidden_layers, config.hidden_width, in_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_every, gamma=0.5)
        if config.lr_decay_every > 0
        else None
    )

    history: dict[str, list[float]] = {
        "total": [], "pde": [], "terminal": [], "boundary_m": [], "boundary_v": [],
    }

    for _ in range(config.epochs):
        optimizer.zero_grad()

        m_int, v_int, tau_int = sample_interior(config.n_interior, config.m_max, config.v_max, config.tau_max)
        residual = pde_residual(
            model, m_int, v_int, tau_int, config.r, config.q, config.kappa, config.theta, config.xi, config.rho
        )
        loss_pde = torch.mean(residual**2)

        m_term, v_term, tau_term = sample_terminal(config.n_terminal, config.m_max, config.v_max)
        u_term_pred = model(torch.cat([m_term, v_term, tau_term], dim=1))
        u_term_true = terminal_condition(m_term, config.option_type)
        loss_terminal = torch.mean((u_term_pred - u_term_true) ** 2)

        v_bnd, tau_bnd = sample_boundary_m(config.n_boundary_m, config.v_max, config.tau_max)
        m_low = torch.zeros_like(v_bnd)
        m_high = torch.full_like(v_bnd, config.m_max)
        u_low_pred = model(torch.cat([m_low, v_bnd, tau_bnd], dim=1))
        u_high_pred = model(torch.cat([m_high, v_bnd, tau_bnd], dim=1))
        u_low_true = boundary_low(tau_bnd, config.r, config.option_type)
        u_high_true = boundary_high(tau_bnd, config.m_max, config.r, config.q, config.option_type)
        loss_boundary_m = torch.mean((u_low_pred - u_low_true) ** 2) + torch.mean((u_high_pred - u_high_true) ** 2)

        m_vmax, tau_vmax = sample_boundary_v_max(config.n_boundary_v, config.m_max, config.tau_max)
        d2u_dv2 = boundary_v_max_residual(model, m_vmax, tau_vmax, config.v_max)
        loss_boundary_v = torch.mean(d2u_dv2**2)

        loss = (
            config.lambda_pde * loss_pde
            + config.lambda_terminal * loss_terminal
            + config.lambda_boundary_m * loss_boundary_m
            + config.lambda_boundary_v * loss_boundary_v
        )
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        history["total"].append(loss.item())
        history["pde"].append(loss_pde.item())
        history["terminal"].append(loss_terminal.item())
        history["boundary_m"].append(loss_boundary_m.item())
        history["boundary_v"].append(loss_boundary_v.item())

    return model, history
