"""v2 options on AmericanPINNConfig (results/uq/paper_v2/DIAGNOSIS.md)."""
import pytest
import torch

from ivuq.pinn import AmericanPINNConfig
from ivuq.pinn.free_boundary import free_boundary_residuals


def test_defaults_are_v1():
    c = AmericanPINNConfig()
    assert c.lcp_form == "product" and c.american_low_boundary is False


def test_bad_lcp_form_rejected():
    with pytest.raises(ValueError):
        AmericanPINNConfig(lcp_form="fb")


class _Const(torch.nn.Module):
    """u(m, tau) = c, so L[u] = r c exactly."""
    def __init__(self, c):
        super().__init__(); self.c = c; self.w = torch.nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.c + 0.0 * (x[:, :1] ** 3 + x[:, 1:2])


def test_min_form_is_min_of_L_and_gap():
    m = torch.tensor([[1.5], [2.0]]); tau = torch.tensor([[0.5], [0.5]])
    r = 0.03
    _, _, prod = free_boundary_residuals(_Const(0.2), m, tau, r, 0.0, 0.2, "put")
    _, _, mn = free_boundary_residuals(_Const(0.2), m, tau, r, 0.0, 0.2, "put", lcp_form="min")
    # h = 0 for m > 1, so u - h = 0.2 and L[u] = r * 0.2 = 0.006
    assert torch.allclose(prod, torch.full_like(prod, 0.006 * 0.2), atol=1e-7)
    assert torch.allclose(mn, torch.full_like(mn, 0.006), atol=1e-7)
