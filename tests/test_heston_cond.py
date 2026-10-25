"""Parameter-conditioned Heston PINNs (`heston_cond.py`, `heston_cond_coupled.py`).

Structural checks only (tiny networks, no real training): the control variate makes the terminal
condition exact for any weights, the BS part matches the closed form, the coupled model's price
respects the payoff and its delta is -1 in the exercise region, and save/load round-trips.
Accuracy against the finite-difference reference is measured by scripts/train_heston_cond.py.
"""

import numpy as np
import torch

from ivuq.pinn.heston_cond import CondNet, HestonCondConfig, HestonCondPINN, ParamBox, bs_put_unit, lcp_terms
from ivuq.pinn.heston_cond_coupled import CoupledConfig, HestonCoupledPINN
from ivuq.pricing import black_scholes

torch.set_num_threads(1)
BOX = ParamBox(r=(0.034, 0.043), theta=(0.01, 0.09), v_max=0.30, q=(0.009, 0.015))
P = np.array([2.0, 0.04, 0.5, -0.6, 0.04, 0.012])


def test_bs_part_matches_closed_form():
    m = torch.tensor([[0.9], [1.0], [1.1]])
    v, tau = torch.full((3, 1), 0.04), torch.full((3, 1), 0.2)
    r, q = torch.full((3, 1), 0.04), torch.full((3, 1), 0.01)
    got = bs_put_unit(m, v, tau, r, q).numpy().ravel()
    ref = [black_scholes.price(100 * x, 100.0, 0.2, 0.04, 0.01, 0.2, "put") / 100 for x in (0.9, 1.0, 1.1)]
    assert np.allclose(got, ref, atol=1e-5)


def test_control_variate_makes_terminal_condition_exact():
    torch.manual_seed(0)
    net = CondNet(BOX, 2, 16, control_variate=True)
    for prm in net.parameters():
        torch.nn.init.normal_(prm, std=1.0)               # arbitrary, large weights
    m = torch.linspace(0.5, 1.6, 9).reshape(-1, 1)
    x = torch.cat([m, torch.full_like(m, 0.04), torch.zeros_like(m), torch.tensor(P, dtype=torch.float32).expand(9, -1)], dim=1)
    u = net(x).detach().numpy().ravel()
    assert np.allclose(u, np.maximum(1 - m.numpy().ravel(), 0.0), atol=2e-5)


def test_pde_residual_is_finite_and_differentiable_in_parameters():
    net = CondNet(BOX, 2, 16, control_variate=True)
    n = 50
    m, v, tau = 0.8 + 0.4 * torch.rand(n, 1), 0.01 + 0.1 * torch.rand(n, 1), 0.02 + 0.2 * torch.rand(n, 1)
    p = torch.tensor(P, dtype=torch.float32).expand(n, -1).clone()
    R, u = lcp_terms(net, m, v, tau, p)
    assert torch.isfinite(R).all() and torch.isfinite(u).all()
    R.pow(2).mean().backward()
    assert all(prm.grad is not None for prm in net.parameters())


def test_cond_roundtrip(tmp_path):
    cfg = HestonCondConfig(box=BOX, hidden_layers=2, hidden_width=16, control_variate=True, epochs=3,
                           n_interior=64, n_terminal=32, n_edge=16)
    a = HestonCondPINN(cfg).fit(log_every=0)
    a.save(tmp_path / "a.pt")
    b = HestonCondPINN.load(tmp_path / "a.pt")
    args = ([0.9, 1.0, 1.1], [0.04] * 3, [0.1] * 3, P)
    assert np.allclose(a.price_u(*args), b.price_u(*args))
    assert np.all(a.price_u(*args) >= np.maximum(1 - np.array([0.9, 1.0, 1.1]), 0) - 1e-6)


def test_coupled_structure_and_roundtrip(tmp_path):
    cfg = CoupledConfig(box=BOX, hidden_layers=2, hidden_width=16, bnd_layers=2, bnd_width=16, epochs=3,
                        n_interior=64, n_bnd=32, n_edge=16)
    a = HestonCoupledPINN(cfg).fit(log_every=0)
    b = a.boundary([0.02, 0.04, 0.1], [0.05, 0.1, 0.2], P)
    assert np.all((b > 0) & (b < 1))
    m = np.linspace(0.7, 1.3, 13)
    u = a.price_u(m, np.full(13, 0.04), np.full(13, 0.1), P)
    assert np.all(u >= np.maximum(1 - m, 0) - 1e-6)           # float32 payoff
    bd = a.boundary([0.04], [0.1], P)[0]
    d = a.delta(100 * m, np.full(13, 100.0), np.full(13, 0.04), np.full(13, 0.1), P)
    assert np.allclose(d[m < bd], -1.0)                       # exercised: payoff slope
    a.save(tmp_path / "c.pt")
    c = HestonCoupledPINN.load(tmp_path / "c.pt")
    assert np.allclose(a.price_u(m, np.full(13, 0.04), np.full(13, 0.1), P), c.price_u(m, np.full(13, 0.04), np.full(13, 0.1), P))
