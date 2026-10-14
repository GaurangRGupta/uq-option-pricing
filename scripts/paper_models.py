"""Load trained paper PINNs and expose them as conformal PointPredictors."""

from __future__ import annotations

import numpy as np
import torch

from ivuq.pinn import AmericanPINN, AmericanPINNConfig, EuropeanPINN, EuropeanPINNConfig
from ivuq.uq import PINNAdapter

from paper_common import PAPER

K = 100.0


def load_pinn(model: str, seed: int):
    ck = torch.load(PAPER / "models" / f"{model}_seed{seed}.pt", weights_only=False)
    if model in ("n1", "n0"):
        pinn = EuropeanPINN(EuropeanPINNConfig(**ck["config"]))
    else:
        pinn = AmericanPINN(AmericanPINNConfig(**ck["config"]))
    from ivuq.pinn.network import PINN

    cfg = pinn.config
    net = PINN(cfg.hidden_layers, cfg.hidden_width)
    net.load_state_dict(ck["state_dict"])
    net.eval()
    pinn.model = net
    return pinn


class LCPAdapter(PINNAdapter):
    """Difficulty = |min(L[u], u - h(m))|, the American put LCP residual."""

    def constraint_residual(self, m, tau):
        m = np.asarray(m, dtype=np.float64).reshape(-1)
        lu = super().constraint_residual(m, tau)
        u = self.predict(m, tau)
        h = np.maximum(1.0 - m, 0.0)
        return np.minimum(lu, u - h)


def predictors(model: str, seed: int) -> dict:
    pinn = load_pinn(model, seed)
    out = {"pde": PINNAdapter(pinn, K=K, name=f"{model}_pde")}
    if model == "n2":
        out["lcp"] = LCPAdapter(pinn, K=K, name=f"{model}_lcp")
    return out
