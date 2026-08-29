"""N0 vs N1 (Phase 2 of RESEARCH_GAP_AND_ROADMAP.md Section 8): does the PDE
residual loss actually make the network learn Black-Scholes, versus a
network fit only to the terminal payoff and the two boundary conditions?

Uses a small network and few epochs (~10s/model on CPU) so this test suite
stays fast. The "real" config used for actually reported numbers is bigger
(more epochs/width) and lives in EuropeanPINNConfig's defaults, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from ivuq.pinn import EuropeanPINN, EuropeanPINNConfig
from ivuq.pricing.black_scholes import price as bs_price

# Small/fast: enough to show the PDE loss matters, not enough to be a
# tight fit. See paper/related_work.md and RESEARCH_GAP_AND_ROADMAP.md
# Section 8 for the bigger, slower config used for reported results.
_FAST_KWARGS = dict(
    r=0.03,
    q=0.0,
    sigma=0.2,
    m_max=3.0,
    tau_max=1.0,
    hidden_layers=3,
    hidden_width=24,
    epochs=1500,
    n_interior=1000,
    n_terminal=200,
    n_boundary=100,
    lr=2e-3,
    lr_decay_every=700,
    seed=0,
)

_K = 100.0
_S_GRID = np.linspace(60.0, 180.0, 25)
_TAU_GRID = np.linspace(0.1, 0.9, 9)


def _grid_mean_abs_error(model: EuropeanPINN, cfg: EuropeanPINNConfig) -> float:
    """Mean |predicted - closed-form BS| over an interior grid, away from the
    domain edges (m near 0 or m_max), where boundary-condition artifacts
    dominate and aren't what this test is checking."""
    SS, TT = np.meshgrid(_S_GRID, _TAU_GRID)
    S_flat, tau_flat = SS.flatten(), TT.flatten()
    pred = model.price(S_flat, _K, tau_flat)
    true = np.array(
        [bs_price(s, _K, t, cfg.r, cfg.q, cfg.sigma, cfg.option_type) for s, t in zip(S_flat, tau_flat)]
    )
    return float(np.mean(np.abs(pred - true)))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_n1_recovers_analytic_black_scholes(option_type: str) -> None:
    cfg = EuropeanPINNConfig(option_type=option_type, lambda_pde=1.0, lambda_terminal=3.0, **_FAST_KWARGS)
    model = EuropeanPINN(cfg).fit()

    mean_err = _grid_mean_abs_error(model, cfg)

    # Loose: this is the fast/small config. Generous but not vacuous -- an
    # untrained or badly broken model misses by 10s of dollars on a $100
    # strike, not single dollars.
    assert mean_err < 5.0, f"N1 ({option_type}) mean abs error {mean_err:.3f} too large vs. closed-form BS"


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_n1_beats_n0_ablation(option_type: str) -> None:
    """N0 (lambda_pde=0) only ever sees the terminal payoff and the two
    boundary conditions -- nothing constrains what it does in between. N1
    adds the PDE residual loss over the interior. If the physics loss is
    doing anything, N1's interior error should be meaningfully lower."""
    n1_cfg = EuropeanPINNConfig(option_type=option_type, lambda_pde=1.0, lambda_terminal=3.0, **_FAST_KWARGS)
    n0_cfg = EuropeanPINNConfig(option_type=option_type, lambda_pde=0.0, lambda_terminal=3.0, **_FAST_KWARGS)

    n1_err = _grid_mean_abs_error(EuropeanPINN(n1_cfg).fit(), n1_cfg)
    n0_err = _grid_mean_abs_error(EuropeanPINN(n0_cfg).fit(), n0_cfg)

    assert n1_err <= 0.95 * n0_err, (
        f"N1 ({option_type}) mean abs error {n1_err:.3f} should be meaningfully below "
        f"N0's {n0_err:.3f} -- the PDE loss isn't showing up as an improvement"
    )
