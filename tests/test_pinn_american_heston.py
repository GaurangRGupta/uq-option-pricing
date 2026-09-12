"""N2-Heston (Phase 3b's centerpiece): does the coupled solution+boundary
network (`heston_free_boundary.py`) recover the American put price, and do
the value-matching/smooth-pasting conditions actually hold at the learned
boundary?

No closed form exists for American options, and Heston has no fast lattice
method the way GBM has CRR, so the reference here is
`ivuq.pricing.heston_lsmc.lsmc_american_heston_price` -- built and
independently sanity-checked in `test_heston_lsmc.py` before this file uses
it as a reference at all, same "validate the instrument first" discipline
as every other stage of this project.

Uses a small network and few epochs so this test suite stays fast. The
"real" config used for actually reported numbers is bigger and lives in
AmericanHestonPINNConfig's defaults, not here.
"""

from __future__ import annotations

import numpy as np
import torch

from ivuq.pinn import AmericanHestonPINN, AmericanHestonPINNConfig
from ivuq.pinn.heston_free_boundary import BoundaryNet, boundary_residuals
from ivuq.pinn.network import PINN
from ivuq.pricing.heston import heston_price
from ivuq.pricing.heston_lsmc import lsmc_american_heston_price

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
    sol_hidden_layers=3,
    sol_hidden_width=32,
    bnd_hidden_layers=2,
    bnd_hidden_width=16,
    epochs=1500,
    n_interior=1500,
    n_boundary_vt=250,
    n_terminal=250,
    n_boundary_m_max=120,
    n_boundary_v_max=120,
    lr=2e-3,
    lr_decay_every=700,
    seed=0,
)

_K = 100.0


def test_n2_heston_recovers_lsmc_price():
    cfg = AmericanHestonPINNConfig(**_FAST_KWARGS)
    model = AmericanHestonPINN(cfg).fit()

    S_grid = np.linspace(70.0, 140.0, 8)
    tau_grid = np.linspace(0.15, 0.85, 4)
    SS, TT = np.meshgrid(S_grid, tau_grid)
    S_flat, tau_flat = SS.flatten(), TT.flatten()

    pred = model.price(S_flat, _K, cfg.v0, tau_flat)
    true = np.array(
        [
            lsmc_american_heston_price(
                s, _K, t, cfg.r, cfg.q, cfg.kappa, cfg.theta, cfg.xi, cfg.rho, cfg.v0,
                "put", n_steps=60, n_paths=8000, seed=0,
            )
            for s, t in zip(S_flat, tau_flat)
        ]
    )
    mean_err = float(np.mean(np.abs(pred - true)))

    # Loose: fast/small config on both sides (PINN and LSMC), and this is a
    # harder target than N1-Heston (a free boundary to locate, not a plain
    # PDE fit). Generous but not vacuous.
    assert mean_err < 8.0, f"N2-Heston mean abs error {mean_err:.3f} too large vs. LSMC"


def test_n2_heston_obeys_obstacle_condition():
    """u >= h(m) (the obstacle inequality `heston_free_boundary.
    obstacle_residual` penalizes) should roughly hold everywhere -- soft
    penalty, so checked on average, same convention as
    `test_pinn_american.test_n2_obeys_obstacle_condition` on the GBM/LCP
    side, not as an exact per-point guarantee."""
    cfg = AmericanHestonPINNConfig(**_FAST_KWARGS)
    model = AmericanHestonPINN(cfg).fit()

    S_grid = np.linspace(50.0, 150.0, 11)
    tau_grid = np.linspace(0.1, 0.9, 5)
    SS, TT = np.meshgrid(S_grid, tau_grid)
    S_flat, tau_flat = SS.flatten(), TT.flatten()

    price = model.price(S_flat, _K, cfg.v0, tau_flat)
    intrinsic = np.maximum(_K - S_flat, 0.0)
    violation = np.maximum(intrinsic - price, 0.0)
    assert np.mean(violation) < 2.0, (
        f"N2-Heston violates the obstacle condition u >= intrinsic by {np.mean(violation):.3f} on average"
    )


def test_n2_heston_boundary_stays_below_strike():
    """b_phi(v, tau) in (0, 1) is enforced by construction (the sigmoid in
    BoundaryNet) -- checks it actually holds after training, at real S*
    units, not just that the constraint exists in the code."""
    cfg = AmericanHestonPINNConfig(**_FAST_KWARGS)
    model = AmericanHestonPINN(cfg).fit()

    v_grid = np.linspace(0.01, 0.15, 5)
    tau_grid = np.linspace(0.05, 0.95, 5)
    VV, TT = np.meshgrid(v_grid, tau_grid)
    b = model.boundary(VV.flatten(), TT.flatten())
    assert np.all(b > 0.0) and np.all(b < 1.0)


def test_n2_heston_at_least_european_heston():
    """American >= European under the same Heston dynamics -- the
    early-exercise right can never make the option less valuable."""
    cfg = AmericanHestonPINNConfig(**_FAST_KWARGS)
    model = AmericanHestonPINN(cfg).fit()

    S_grid = np.linspace(70.0, 140.0, 8)
    tau_grid = np.linspace(0.15, 0.85, 4)
    SS, TT = np.meshgrid(S_grid, tau_grid)
    S_flat, tau_flat = SS.flatten(), TT.flatten()

    american = model.price(S_flat, _K, cfg.v0, tau_flat)
    european = np.array(
        [
            heston_price(s, _K, t, cfg.r, cfg.q, cfg.kappa, cfg.theta, cfg.xi, cfg.rho, cfg.v0, "put")
            for s, t in zip(S_flat, tau_flat)
        ]
    )
    # Loose slack -- this is a soft-trained network, not an exact solve.
    assert np.mean(american - european) > -1.5


def test_value_matching_and_smooth_pasting_residuals_small_after_training():
    """Directly checks (5)/(6) in heston_free_boundary.py at the trained
    boundary network's own points -- the actual free-boundary conditions,
    not just the resulting price accuracy."""
    cfg = AmericanHestonPINNConfig(**_FAST_KWARGS)
    model = AmericanHestonPINN(cfg).fit()

    v = torch.rand(200, 1) * cfg.v_max
    tau = torch.rand(200, 1) * cfg.tau_max
    value_matching, smooth_pasting, _ = boundary_residuals(model.sol_net, model.bnd_net, v, tau)

    assert torch.mean(value_matching**2).item() < 0.05
    assert torch.mean(smooth_pasting**2).item() < 0.3


def test_boundary_net_output_always_in_unit_interval():
    """Math/code-consistency check on BoundaryNet itself, independent of
    training: an untrained network's sigmoid output must still lie in
    (0, 1) for arbitrary (v, tau)."""
    torch.manual_seed(0)
    bnd_net = BoundaryNet(hidden_layers=2, hidden_width=8)
    v = torch.rand(500, 1) * 0.5
    tau = torch.rand(500, 1) * 2.0
    b = bnd_net(v, tau)
    assert torch.all(b > 0.0) and torch.all(b < 1.0)
