"""Split conformal: quantile formula, coverage on a known-noise sandbox, Mondrian."""

from __future__ import annotations

import numpy as np
import pytest

from ivuq.uq.conformal import (
    auto_kappa,
    conformal_quantile,
    empirical_coverage,
    mean_width,
    mondrian_split_conformal_interval,
    moneyness_tenor_buckets,
    nonconformity_scores,
    split_conformal_interval,
)


class _OracleMean:
    """predict = true mean; labels are mean + noise. Residual = |m-1|."""

    name = "oracle"

    def __init__(self, mean_fn, difficulty_fn=None) -> None:
        self.mean_fn = mean_fn
        self.difficulty_fn = difficulty_fn

    def predict(self, m, tau):
        return self.mean_fn(np.asarray(m), np.asarray(tau))

    def constraint_residual(self, m, tau):
        if self.difficulty_fn is None:
            return None
        return self.difficulty_fn(np.asarray(m), np.asarray(tau))


def test_conformal_quantile_matches_order_statistic():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # n=10, alpha=0.1 -> k = ceil(11*0.9) = 10 -> 10th order stat = 1.0
    assert conformal_quantile(scores, 0.1) == pytest.approx(1.0)
    # n=10, alpha=0.2 -> k = ceil(11*0.8) = 9 -> 0.9
    assert conformal_quantile(scores, 0.2) == pytest.approx(0.9)


def test_conformal_quantile_is_infinite_when_k_exceeds_n():
    scores = np.array([1.0, 2.0])
    # n=2, alpha=0.1 -> k = ceil(3*0.9) = 3 > 2
    assert conformal_quantile(scores, 0.1) == float("inf")


def test_homoscedastic_oracle_covers_near_nominal():
    rng = np.random.default_rng(0)
    n_cal, n_test = 400, 2000
    m_cal = rng.uniform(0.6, 1.4, n_cal)
    tau_cal = rng.uniform(0.1, 1.0, n_cal)
    m_test = rng.uniform(0.6, 1.4, n_test)
    tau_test = rng.uniform(0.1, 1.0, n_test)

    def mean_fn(m, tau):
        return 0.1 + 0.2 * np.maximum(1.0 - m, 0.0) * np.sqrt(tau)

    pred = _OracleMean(mean_fn)
    noise = 0.03
    u_cal = mean_fn(m_cal, tau_cal) + rng.normal(0.0, noise, n_cal)
    u_test = mean_fn(m_test, tau_test) + rng.normal(0.0, noise, n_test)
    result = split_conformal_interval(
        pred, m_cal, tau_cal, u_cal, m_test, tau_test, alpha=0.1, kappa=0.0
    )
    cov = empirical_coverage(result.lo, result.hi, u_test)
    assert 0.86 <= cov <= 0.96
    assert result.kappa == 0.0
    assert np.all(np.isfinite(result.lo))


def test_normalised_scores_widen_where_difficulty_is_large():
    rng = np.random.default_rng(1)
    n = 300
    m = rng.uniform(0.5, 1.5, n)
    tau = rng.uniform(0.1, 1.0, n)
    u_hat = np.zeros(n)
    u = np.zeros(n)
    difficulty = np.abs(m - 1.0)
    vanilla = nonconformity_scores(u, u_hat, difficulty, kappa=0.0)
    scaled = nonconformity_scores(u, u_hat, difficulty, kappa=10.0)
    # zero errors -> both zero; use a constant error instead
    u = np.full(n, 0.2)
    vanilla = nonconformity_scores(u, u_hat, difficulty, kappa=0.0)
    scaled = nonconformity_scores(u, u_hat, difficulty, kappa=10.0)
    assert np.allclose(vanilla, 0.2)
    assert scaled[np.argmax(difficulty)] < scaled[np.argmin(difficulty)]


def test_mondrian_uses_a_quantile_per_bucket():
    rng = np.random.default_rng(2)
    n_cal, n_test = 600, 300
    m_cal = rng.uniform(0.5, 1.5, n_cal)
    tau_cal = rng.uniform(0.05, 1.0, n_cal)
    m_test = rng.uniform(0.5, 1.5, n_test)
    tau_test = rng.uniform(0.05, 1.0, n_test)

    def mean_fn(m, tau):
        return 0.05 * np.ones_like(m)

    pred = _OracleMean(mean_fn)
    u_cal = mean_fn(m_cal, tau_cal) + rng.normal(0.0, 0.02, n_cal)
    result = mondrian_split_conformal_interval(
        pred, m_cal, tau_cal, u_cal, m_test, tau_test, alpha=0.1
    )
    assert isinstance(result.q, dict)
    assert len(result.q) >= 1
    buckets = moneyness_tenor_buckets(m_test, tau_test)
    for b, q in result.q.items():
        mask = buckets == b
        if mask.any() and np.isfinite(q):
            widths = result.hi[mask] - result.lo[mask]
            assert np.allclose(widths, 2 * q)


def test_auto_kappa_is_reciprocal_median():
    d = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    assert auto_kappa(d) == pytest.approx(1.0 / 2.0)


def test_mean_width_and_coverage_helpers():
    lo = np.array([0.0, 0.0])
    hi = np.array([2.0, 4.0])
    u = np.array([1.0, 5.0])
    assert mean_width(lo, hi) == pytest.approx(3.0)
    assert empirical_coverage(lo, hi, u) == pytest.approx(0.5)
