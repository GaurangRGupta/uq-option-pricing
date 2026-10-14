"""The UQ adapters must match PointPredictor without training a real PINN."""

from __future__ import annotations

import numpy as np

from ivuq.uq import BaselineAdapter, PINNAdapter, PointPredictor, split_conformal_interval


class _FakePinn:
    def price(self, S, K, tau):
        S = np.atleast_1d(np.asarray(S, dtype=np.float64)).reshape(-1)
        return np.full(S.shape, 0.1 * float(np.mean(np.atleast_1d(K))))

    def pde_residual(self, S, K, tau):
        n = np.atleast_1d(np.asarray(S)).reshape(-1).shape[0]
        return np.full(n, 0.02)


class _FakeBaseline:
    name = "fake_ml"

    def predict(self, X):
        return np.full(len(X), 0.3)


def test_pinn_adapter_is_a_point_predictor():
    adapter = PINNAdapter(_FakePinn(), K=100.0, name="fake_pinn")
    assert isinstance(adapter, PointPredictor)
    m = np.array([0.8, 1.0, 1.2])
    tau = np.array([0.2, 0.5, 0.8])
    u = adapter.predict(m, tau)
    r = adapter.constraint_residual(m, tau)
    assert u.shape == (3,)
    assert np.allclose(u, 0.1)
    assert np.allclose(r, 0.02)


def test_baseline_adapter_has_no_physics_residual():
    adapter = BaselineAdapter(_FakeBaseline())
    assert isinstance(adapter, PointPredictor)
    m = np.array([1.0, 1.1])
    tau = np.array([0.4, 0.6])
    assert np.allclose(adapter.predict(m, tau), 0.3)
    assert adapter.constraint_residual(m, tau) is None


def test_split_conformal_runs_on_the_pinn_adapter():
    adapter = PINNAdapter(_FakePinn(), K=100.0, name="fake_pinn")
    m = np.array([0.8, 1.0, 1.2, 0.9, 1.1])
    tau = np.array([0.2, 0.5, 0.8, 0.3, 0.4])
    u = adapter.predict(m, tau) + np.array([0.01, -0.02, 0.0, 0.03, -0.01])
    result = split_conformal_interval(adapter, m, tau, u, m, tau, alpha=0.2)
    assert result.u_hat.shape == (5,)
    assert result.lo.shape == (5,)
    assert np.all(result.hi >= result.lo)
