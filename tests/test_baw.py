import pytest

from ivuq.pricing import black_scholes as bs
from ivuq.pricing.baw import baw_price
from ivuq.pricing.binomial import crr_price


def test_baw_call_no_dividend_matches_european_exactly():
    S, K, T, r, q, sigma = 100.0, 95.0, 0.75, 0.05, 0.0, 0.3
    baw = baw_price(S, K, T, r, q, sigma, "call")
    european = bs.price(S, K, T, r, q, sigma, "call")
    assert baw == pytest.approx(european, abs=1e-9)


@pytest.mark.parametrize(
    "S,K,T,r,q,sigma,option_type",
    [
        (100.0, 100.0, 1.0, 0.05, 0.03, 0.3, "call"),
        (100.0, 100.0, 1.0, 0.05, 0.03, 0.3, "put"),
        (80.0, 100.0, 0.5, 0.03, 0.0, 0.25, "put"),
        (120.0, 100.0, 0.5, 0.04, 0.06, 0.2, "call"),
        (40.0, 100.0, 1.0, 0.08, 0.0, 0.2, "put"),
    ],
)
def test_baw_agrees_with_tree_within_tolerance(S, K, T, r, q, sigma, option_type):
    baw = baw_price(S, K, T, r, q, sigma, option_type)
    tree = crr_price(S, K, T, r, q, sigma, option_type, is_american=True, steps=1000)
    # BAW is an approximation used only to warm-start the solver; a few percent
    # relative error against the tree (the reference pricer) is expected and fine.
    assert baw == pytest.approx(tree, rel=0.05, abs=0.05)
