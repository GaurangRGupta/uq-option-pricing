import pytest

from ivuq.pricing import black_scholes as bs
from ivuq.pricing.binomial import crr_price


def test_european_tree_converges_to_black_scholes():
    S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.04, 0.01, 0.25
    bs_call = bs.price(S, K, T, r, q, sigma, "call")
    tree_call = crr_price(S, K, T, r, q, sigma, "call", is_american=False, steps=2000)
    assert tree_call == pytest.approx(bs_call, abs=5e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_american_at_least_as_valuable_as_european(option_type):
    S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.03, 0.3
    european = crr_price(S, K, T, r, q, sigma, option_type, is_american=False, steps=300)
    american = crr_price(S, K, T, r, q, sigma, option_type, is_american=True, steps=300)
    assert american >= european - 1e-9


def test_zero_dividend_american_call_equals_european_call():
    # Never optimal to exercise a call early with no dividends: this must hold exactly
    # in a correct tree, up to discretization noise.
    S, K, T, r, q, sigma = 100.0, 95.0, 0.75, 0.05, 0.0, 0.3
    european = crr_price(S, K, T, r, q, sigma, "call", is_american=False, steps=500)
    american = crr_price(S, K, T, r, q, sigma, "call", is_american=True, steps=500)
    assert american == pytest.approx(european, abs=1e-6)


def test_american_put_at_least_intrinsic():
    S, K, T, r, q, sigma = 80.0, 100.0, 1.0, 0.05, 0.0, 0.3
    american_put = crr_price(S, K, T, r, q, sigma, "put", is_american=True, steps=300)
    assert american_put >= max(K - S, 0.0) - 1e-9


def test_deep_itm_american_put_exercises_early():
    # Deep ITM American put on a low/no-growth stock should be worth more than
    # its European counterpart by a nontrivial margin (early exercise has real value).
    S, K, T, r, q, sigma = 40.0, 100.0, 1.0, 0.08, 0.0, 0.2
    european = crr_price(S, K, T, r, q, sigma, "put", is_american=False, steps=500)
    american = crr_price(S, K, T, r, q, sigma, "put", is_american=True, steps=500)
    assert american > european + 0.5


def test_rejects_bad_option_type():
    with pytest.raises(ValueError):
        crr_price(100, 100, 1, 0.03, 0.0, 0.2, "strangle", is_american=True)
