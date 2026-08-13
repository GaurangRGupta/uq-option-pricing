import pytest

from ivuq.pricing.binomial import crr_price
from ivuq.pricing.iv_solver import implied_vol


def test_european_round_trip_matches_black_scholes_solver():
    S, K, T, r, q, sigma_true = 100.0, 105.0, 0.6, 0.04, 0.015, 0.28
    from ivuq.pricing import black_scholes as bs

    price = bs.price(S, K, T, r, q, sigma_true, "call")
    recovered = implied_vol(price, S, K, T, r, q, "call", is_american=False)
    assert recovered == pytest.approx(sigma_true, abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_american_round_trip(option_type):
    S, K, T, r, q, sigma_true = 100.0, 100.0, 0.5, 0.05, 0.02, 0.3
    price = crr_price(S, K, T, r, q, sigma_true, option_type, is_american=True, steps=400)
    recovered = implied_vol(price, S, K, T, r, q, option_type, is_american=True, steps=400)
    assert recovered == pytest.approx(sigma_true, abs=5e-3)
