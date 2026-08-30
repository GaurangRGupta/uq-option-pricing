"""N0/N1-Heston (Phase 3b, first step): does the Heston PDE residual loss
make the network learn Heston's own dynamics?

Two checks, both against closed forms rather than the fast-config price
grid alone: (a) equation (4) in heston_pde.py collapses onto equation (2)
in black_scholes_pde.py when xi=0 and v is held at sigma^2 -- checked
directly on the residual formula, not just on a trained network, so this
is really a math/code-consistency check on `heston_pde.py` itself; (b) N1-
Heston recovers the closed-form `heston_price` (ivuq.pricing.heston),
analogous to how N1 recovers closed-form Black-Scholes.

Uses a small network and few epochs so this test suite stays fast. The
"real" config used for actually reported numbers is bigger and lives in
HestonEuropeanPINNConfig's defaults, not here.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ivuq.pinn import EuropeanHestonPINN, HestonEuropeanPINNConfig
from ivuq.pinn.black_scholes_pde import pde_residual as bs_pde_residual
from ivuq.pinn.heston_pde import pde_residual as heston_pde_residual
from ivuq.pinn.network import PINN
from ivuq.pricing.heston import heston_price

_FAST_KWARGS = dict(
    r=0.03,
    q=0.0,
    kappa=2.0,
    theta=0.04,
    xi=0.4,
    rho=-0.5,
    v0=0.04,
    m_max=3.0,
    v_max=0.16,
    tau_max=1.0,
    hidden_layers=3,
    hidden_width=32,
    epochs=1500,
    n_interior=1500,
    n_terminal=250,
    n_boundary_m=120,
    n_boundary_v=120,
    lr=2e-3,
    lr_decay_every=700,
    seed=0,
)

_K = 100.0
_S_GRID = np.linspace(70.0, 160.0, 10)
_TAU_GRID = np.linspace(0.15, 0.85, 5)


def test_heston_pde_collapses_onto_black_scholes_pde_at_zero_vol_of_vol() -> None:
    """Direct check on heston_pde.pde_residual itself (equation (4)): with
    xi=0 and v held fixed at sigma^2 for every point, it should return
    exactly the same residual as black_scholes_pde.pde_residual (equation
    (2)) for the same network, up to floating-point noise -- this is the
    math/code-consistency check for heston_pde.py, independent of training."""
    torch.manual_seed(0)
    model_2d = PINN(hidden_layers=2, hidden_width=8, in_dim=2)

    n = 200
    sigma = 0.2
    m = torch.rand(n, 1) * 3.0
    tau = torch.rand(n, 1) * 1.0
    v = torch.full_like(m, sigma**2)
    r, q = 0.03, 0.0

    bs_residual = bs_pde_residual(model_2d, m, tau, r, q, sigma)

    # Wrap the 2D model so it can be called with a 3-column (m, v, tau) input
    # (v ignored) -- lets heston_pde_residual differentiate through the same
    # underlying function without a v dependence, which is exactly the
    # xi=0 / fixed-v limit.
    def model_3d(x: torch.Tensor) -> torch.Tensor:
        return model_2d(torch.cat([x[:, 0:1], x[:, 2:3]], dim=1))

    heston_residual = heston_pde_residual(
        model_3d, m, v, tau, r=r, q=q, kappa=2.0, theta=sigma**2, xi=0.0, rho=-0.5
    )

    assert torch.allclose(heston_residual, bs_residual, atol=1e-5), (
        "Heston PDE residual should collapse onto the Black-Scholes residual when xi=0 and v=sigma^2"
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_n1_heston_recovers_closed_form_heston_price(option_type: str) -> None:
    cfg = HestonEuropeanPINNConfig(option_type=option_type, lambda_pde=1.0, lambda_terminal=3.0, **_FAST_KWARGS)
    model = EuropeanHestonPINN(cfg).fit()

    SS, TT = np.meshgrid(_S_GRID, _TAU_GRID)
    S_flat, tau_flat = SS.flatten(), TT.flatten()
    pred = model.price(S_flat, _K, cfg.v0, tau_flat)
    true = np.array(
        [
            heston_price(s, _K, t, cfg.r, cfg.q, cfg.kappa, cfg.theta, cfg.xi, cfg.rho, cfg.v0, option_type)
            for s, t in zip(S_flat, tau_flat)
        ]
    )
    mean_err = float(np.mean(np.abs(pred - true)))

    # Loose: fast/small config, and Heston has one more state variable to
    # resolve than plain BS. Generous but not vacuous.
    assert mean_err < 6.0, f"N1-Heston ({option_type}) mean abs error {mean_err:.3f} too large vs. closed-form Heston"


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_n1_heston_beats_n0_heston_ablation(option_type: str) -> None:
    """Same ablation as N0-vs-N1 under GBM: N0-Heston (lambda_pde=0) only
    ever sees the terminal payoff and the three boundary conditions --
    nothing constrains the interior. If the Heston PDE loss is doing
    anything, N1-Heston's interior error should be meaningfully lower."""
    n1_cfg = HestonEuropeanPINNConfig(option_type=option_type, lambda_pde=1.0, lambda_terminal=3.0, **_FAST_KWARGS)
    n0_cfg = HestonEuropeanPINNConfig(option_type=option_type, lambda_pde=0.0, lambda_terminal=3.0, **_FAST_KWARGS)

    SS, TT = np.meshgrid(_S_GRID, _TAU_GRID)
    S_flat, tau_flat = SS.flatten(), TT.flatten()
    true = np.array(
        [
            heston_price(s, _K, t, n1_cfg.r, n1_cfg.q, n1_cfg.kappa, n1_cfg.theta, n1_cfg.xi, n1_cfg.rho, n1_cfg.v0, option_type)
            for s, t in zip(S_flat, tau_flat)
        ]
    )

    n1_pred = EuropeanHestonPINN(n1_cfg).fit().price(S_flat, _K, n1_cfg.v0, tau_flat)
    n0_pred = EuropeanHestonPINN(n0_cfg).fit().price(S_flat, _K, n0_cfg.v0, tau_flat)
    n1_err = float(np.mean(np.abs(n1_pred - true)))
    n0_err = float(np.mean(np.abs(n0_pred - true)))

    assert n1_err <= 0.95 * n0_err, (
        f"N1-Heston ({option_type}) mean abs error {n1_err:.3f} should be meaningfully below "
        f"N0-Heston's {n0_err:.3f} -- the PDE loss isn't showing up as an improvement"
    )
