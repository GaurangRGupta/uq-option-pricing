import numpy as np
import pytest

from ivuq.pricing import black_scholes as bs


def test_put_call_parity():
    S, K, T, r, q, sigma = 100.0, 95.0, 0.75, 0.03, 0.01, 0.25
    call = bs.price(S, K, T, r, q, sigma, "call")
    put = bs.price(S, K, T, r, q, sigma, "put")
    parity_rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert call - put == pytest.approx(parity_rhs, abs=1e-10)


def test_hull_textbook_example():
    # Hull, "Options, Futures and Other Derivatives": S=42, K=40, T=0.5, r=10%, sigma=20%, no dividend.
    # Loose tolerance: this is a sanity check against a well-known worked example, not a golden reference.
    S, K, T, r, q, sigma = 42.0, 40.0, 0.5, 0.10, 0.0, 0.20
    call = bs.price(S, K, T, r, q, sigma, "call")
    put = bs.price(S, K, T, r, q, sigma, "put")
    assert call == pytest.approx(4.76, abs=0.02)
    assert put == pytest.approx(0.81, abs=0.02)


def test_at_expiry_equals_intrinsic():
    S, K, r, q, sigma = 100.0, 90.0, 0.05, 0.02, 0.3
    assert bs.price(S, K, 1e-12, r, q, sigma, "call") == pytest.approx(max(S - K, 0), abs=1e-6)
    assert bs.price(S, K, 1e-12, r, q, sigma, "put") == pytest.approx(max(K - S, 0), abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_matches_finite_difference(option_type):
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.03, 0.01, 0.2
    h = 1e-4
    fd_delta = (
        bs.price(S + h, K, T, r, q, sigma, option_type) - bs.price(S - h, K, T, r, q, sigma, option_type)
    ) / (2 * h)
    analytic = bs.delta(S, K, T, r, q, sigma, option_type)
    assert analytic == pytest.approx(fd_delta, abs=1e-4)


def test_vega_matches_finite_difference():
    S, K, T, r, q, sigma = 100.0, 105.0, 0.4, 0.02, 0.0, 0.35
    h = 1e-5
    fd_vega = (bs.price(S, K, T, r, q, sigma + h, "call") - bs.price(S, K, T, r, q, sigma - h, "call")) / (2 * h)
    analytic = bs.vega(S, K, T, r, q, sigma)
    assert analytic == pytest.approx(fd_vega, abs=1e-3)


def test_deep_itm_call_converges_to_discounted_forward_intrinsic():
    S, K, T, r, q, sigma = 1000.0, 10.0, 1.0, 0.05, 0.0, 0.2
    call = bs.price(S, K, T, r, q, sigma, "call")
    intrinsic = S * np.exp(-q * T) - K * np.exp(-r * T)
    assert call == pytest.approx(intrinsic, rel=1e-3)


def test_deep_otm_put_near_zero():
    S, K, T, r, q, sigma = 1000.0, 10.0, 1.0, 0.05, 0.0, 0.2
    put = bs.price(S, K, T, r, q, sigma, "put")
    assert put < 1e-6


def test_implied_vol_round_trip():
    S, K, T, r, q, sigma_true = 100.0, 105.0, 0.6, 0.04, 0.015, 0.28
    for option_type in ("call", "put"):
        p = bs.price(S, K, T, r, q, sigma_true, option_type)
        recovered = bs.implied_vol(p, S, K, T, r, q, option_type)
        assert recovered == pytest.approx(sigma_true, abs=1e-6)


def test_implied_vol_rejects_price_outside_bounds():
    S, K, T, r, q = 100.0, 100.0, 1.0, 0.03, 0.0
    with pytest.raises(ValueError):
        bs.implied_vol(market_price=S + 50, S=S, K=K, T=T, r=r, q=q, option_type="call")


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs.price(100, 100, 1, 0.03, 0.0, 0.2, "straddle")
