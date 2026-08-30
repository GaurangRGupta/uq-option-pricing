"""N2 (Phase 3 of RESEARCH_GAP_AND_ROADMAP.md Section 8): does the
free-boundary PINN recover the American price, and do all three
loss-weighting schemes (fixed/curriculum/self_adaptive) actually converge?

No closed form exists for American options, so the reference here is the
CRR binomial tree (`ivuq.pricing.binomial.crr_price`), the project's
existing American workhorse pricer -- comparing against it is itself new
relative to the reviewed papers (see RESEARCH_GAP_AND_ROADMAP.md Section 2),
none of which check their American PINN against a tree.

Uses a small network and few epochs so this test suite stays fast. The
"real" config used for actually reported numbers is bigger and lives in
AmericanPINNConfig's defaults, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from ivuq.pinn import AmericanPINN, AmericanPINNConfig
from ivuq.pricing.binomial import crr_price

# Small/fast: enough to show each scheme converges, not enough to be a
# tight fit. See RESEARCH_GAP_AND_ROADMAP.md Section 8 for the bigger,
# slower config used for reported results.
_FAST_KWARGS = dict(
    r=0.03,
    q=0.0,
    sigma=0.2,
    m_max=3.0,
    tau_max=1.0,
    hidden_layers=3,
    hidden_width=32,
    epochs=1200,
    n_interior=800,
    n_terminal=150,
    n_boundary=80,
    lr=3e-3,
    lr_decay_every=600,
    curriculum_ramp_epochs=400,
    seed=0,
)

_K = 100.0
_S_GRID = np.linspace(60.0, 180.0, 13)
_TAU_GRID = np.linspace(0.1, 0.9, 5)


def _grid_mean_abs_error(model: AmericanPINN, cfg: AmericanPINNConfig) -> float:
    """Mean |predicted - CRR reference| over an interior grid, away from the
    domain edges (m near 0 or m_max), where boundary-condition artifacts
    dominate and aren't what this test is checking."""
    SS, TT = np.meshgrid(_S_GRID, _TAU_GRID)
    S_flat, tau_flat = SS.flatten(), TT.flatten()
    pred = model.price(S_flat, _K, tau_flat)
    true = np.array(
        [
            crr_price(s, _K, t, cfg.r, cfg.q, cfg.sigma, cfg.option_type, is_american=True)
            for s, t in zip(S_flat, tau_flat)
        ]
    )
    return float(np.mean(np.abs(pred - true)))


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("weighting_scheme", ["fixed", "curriculum", "self_adaptive"])
def test_n2_recovers_crr_price(option_type: str, weighting_scheme: str) -> None:
    cfg = AmericanPINNConfig(option_type=option_type, weighting_scheme=weighting_scheme, **_FAST_KWARGS)
    model = AmericanPINN(cfg).fit()

    mean_err = _grid_mean_abs_error(model, cfg)

    # Loose: this is the fast/small config, and the LCP is a harder target
    # than plain BS (three extra penalty terms fighting the data-fit terms).
    # Generous but not vacuous -- an untrained or badly broken model misses
    # by 10s of dollars on a $100 strike, not single dollars.
    assert mean_err < 8.0, (
        f"N2 ({option_type}, {weighting_scheme}) mean abs error {mean_err:.3f} too large vs. CRR"
    )


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_n2_obeys_obstacle_condition(option_type: str) -> None:
    """u >= h(m) (equation (3) in free_boundary.py) should hold everywhere,
    up to a small tolerance for the soft penalty not being exactly zero."""
    cfg = AmericanPINNConfig(option_type=option_type, weighting_scheme="fixed", **_FAST_KWARGS)
    model = AmericanPINN(cfg).fit()

    S_flat = np.repeat(_S_GRID, _TAU_GRID.shape[0])
    tau_flat = np.tile(_TAU_GRID, _S_GRID.shape[0])
    price = model.price(S_flat, _K, tau_flat)
    intrinsic = np.maximum(S_flat - _K, 0.0) if option_type == "call" else np.maximum(_K - S_flat, 0.0)

    violation = np.maximum(intrinsic - price, 0.0)
    assert np.mean(violation) < 2.0, (
        f"N2 ({option_type}) violates the obstacle condition u >= intrinsic by {np.mean(violation):.3f} on average"
    )
