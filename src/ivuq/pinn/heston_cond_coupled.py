"""Parameter-conditioned coupled solution + boundary PINN for the Heston American put.

Same idea as `heston_free_boundary.py` (a solution network that only has to be right
where holding is optimal, plus an exercise-boundary network tied to it by value
matching and smooth pasting), extended so that the Heston parameters are network inputs
(`heston_cond.py`) and with the Black-Scholes control variate. The point of the split is the
one the LCP min-form of `heston_cond.py` could not deliver: the solution network is never
asked to satisfy an inequality, only the PDE, so there is no supersolution bias to push it
upward.

Unknowns, with p = (kappa, theta, xi, rho, r, q), m = S/K, u = V/K:
    u(m, v, tau, p)   solution network, valid in the continuation region m > b
                      u = BS(m, sigma^2 = v, tau) + (tau / tau_max) N(m, v, tau, p)        (1)
    b(v, tau, p)      exercise boundary network, b in (0, 1) through a sigmoid
The price is  u_am = u  for m >= b,  h(m) = 1 - m  for m < b.

Conditions (R is the Heston operator of eq. (1) in `heston_cond.py`):
    R[u] = 0                                   m in (b, m_hi)      PDE in the continuation region     (2)
    u(b, v, tau, p) = 1 - b                                         value matching                    (3)
    u_m(b, v, tau, p) = -1                                          smooth pasting                    (4)
    b(v, 0, p) = 1                                                  boundary at expiry                (5)
    u(m_hi, v, tau, p) = 0 ;  u_vv(m, v_max, tau, p) = 0                                              (6)
    u >= h in the continuation region                                                                (7)
(1) already gives u(m, v, 0, p) = h(m), so there is no separate terminal term.

Loss: mean squares of (2) with m drawn above the current b (b detached, power-law
concentration just above it), (3) and (4) at m = b(v, tau, p) so that gradients flow into both
networks, (5), (6), the obstacle violation relu(h - u) from (7), and two optional anchor terms
when FD labels are supplied: u against the finite-difference price at continuation points, and
b against the FD exercise boundary. With the anchors off this is a pure PINN.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .heston_cond import (CondNet, ParamBox, _sample_tau, _sample_v, _vmax_term, intrinsic, lcp_terms,
                          sample_params)

__all__ = ["CoupledConfig", "BoundaryCondNet", "HestonCoupledPINN"]


@dataclass
class CoupledConfig:
    box: ParamBox = field(default_factory=ParamBox)
    hidden_layers: int = 5
    hidden_width: int = 128
    bnd_layers: int = 4
    bnd_width: int = 64
    n_interior: int = 4000
    n_bnd: int = 600
    n_edge: int = 300
    n_data: int = 6000
    n_bdata: int = 1500
    w_pde: float = 1.0
    w_obs: float = 5.0
    w_vm: float = 5.0
    w_sp: float = 1.0
    w_tb: float = 5.0
    w_high: float = 1.0
    w_vmax: float = 0.1
    w_data: float = 0.0
    w_bdata: float = 0.0
    lr: float = 2e-3
    lr_final: float = 5e-5
    epochs: int = 6000
    seed: int = 0


class BoundaryCondNet(nn.Module):
    """b(v, tau, p) in (0, 1); the last bias starts so that b is about 0.9 at initialisation."""

    def __init__(self, box: ParamBox, layers: int, width: int) -> None:
        super().__init__()
        plo, phi = box.lo_hi()
        lo = np.concatenate([[0.0, 0.0], plo])
        hi = np.concatenate([[box.v_max, box.tau_max], phi])
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))
        mods: list[nn.Module] = []
        d = 8
        for _ in range(layers):
            mods += [nn.Linear(d, width), nn.Tanh()]
            d = width
        last = nn.Linear(d, 1)
        nn.init.normal_(last.weight, std=0.01)
        nn.init.constant_(last.bias, float(np.log(0.9 / 0.1)))
        mods.append(last)
        self.net = nn.Sequential(*mods)

    def forward(self, v, tau, p):
        x = torch.cat([v, tau, p], dim=1)
        return torch.sigmoid(self.net(2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0))


def _m_above_boundary(b: torch.Tensor, m_hi: float) -> torch.Tensor:
    """Half the points concentrated just above b (power 3), half uniform on (b, m_hi)."""
    n = b.shape[0]
    u = torch.cat([torch.rand(n // 2, 1) ** 3, torch.rand(n - n // 2, 1)], dim=0)
    return b + u * (m_hi - b)


def _losses(sol: CondNet, bnd: BoundaryCondNet, cfg: CoupledConfig, data):
    box = cfg.box
    n = cfg.n_interior
    p = sample_params(n, box)
    v, tau = _sample_v(p, box), _sample_tau(n, box)
    with torch.no_grad():
        b0 = bnd(v, tau, p)
    m = _m_above_boundary(b0, box.m_hi)
    R, u = lcp_terms(sol, m, v, tau, p)
    L = {"pde": torch.mean(R**2), "obs": torch.mean(torch.relu(intrinsic(m) - u) ** 2)}

    nb = cfg.n_bnd
    pb = sample_params(nb, box)
    vb, taub = _sample_v(pb, box), _sample_tau(nb, box)
    b = bnd(vb, taub, pb)
    bq = b.clone().requires_grad_(True) if not b.requires_grad else b
    ub = sol(torch.cat([bq, vb, taub, pb], dim=1))
    dub = torch.autograd.grad(ub, bq, grad_outputs=torch.ones_like(ub), create_graph=True)[0]
    L["vm"] = torch.mean((ub - (1.0 - bq)) ** 2)
    L["sp"] = torch.mean((dub + 1.0) ** 2)
    pt = sample_params(nb, box)
    L["tb"] = torch.mean((bnd(_sample_v(pt, box), torch.zeros(nb, 1), pt) - 1.0) ** 2)

    ne = cfg.n_edge
    pe = sample_params(ne, box)
    ve, taue = _sample_v(pe, box), _sample_tau(ne, box)
    uh = sol(torch.cat([torch.full((ne, 1), box.m_hi), ve, taue, pe], dim=1))
    L["high"] = torch.mean(uh**2)
    pv = sample_params(ne, box)
    mv = 0.9 + (box.m_hi - 0.9) * torch.rand(ne, 1)
    L["vmax"] = torch.mean(_vmax_term(sol, mv, _sample_tau(ne, box), pv, box.v_max) ** 2)

    if data is not None:
        Xu, yu, Xb, yb = data
        if cfg.w_data > 0:
            i = torch.randint(0, Xu.shape[0], (cfg.n_data,))
            L["data"] = torch.mean((sol(Xu[i]) - yu[i]) ** 2)
        if cfg.w_bdata > 0:
            j = torch.randint(0, Xb.shape[0], (cfg.n_bdata,))
            bpred = bnd(Xb[j, 0:1], Xb[j, 1:2], Xb[j, 2:])
            L["bdata"] = torch.mean((bpred - yb[j]) ** 2)
    return L


def _total(cfg: CoupledConfig, L):
    t = (cfg.w_pde * L["pde"] + cfg.w_obs * L["obs"] + cfg.w_vm * L["vm"] + cfg.w_sp * L["sp"]
         + cfg.w_tb * L["tb"] + cfg.w_high * L["high"] + cfg.w_vmax * L["vmax"])
    if "data" in L:
        t = t + cfg.w_data * L["data"]
    if "bdata" in L:
        t = t + cfg.w_bdata * L["bdata"]
    return t


class HestonCoupledPINN:
    """Same inference interface as `HestonCondPINN` (price_u, price, delta, config.box), so the
    calibration code can use either."""

    def __init__(self, config: CoupledConfig) -> None:
        self.config = config
        self.sol = CondNet(config.box, config.hidden_layers, config.hidden_width, control_variate=True)
        self.bnd = BoundaryCondNet(config.box, config.bnd_layers, config.bnd_width)
        self.history: dict[str, list[float]] = {}

    def fit(self, data=None, log_every: int = 250, ckpt: str | Path | None = None):
        cfg = self.config
        torch.manual_seed(cfg.seed)
        params = list(self.sol.parameters()) + list(self.bnd.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=(cfg.lr_final / cfg.lr) ** (1.0 / max(cfg.epochs, 1)))
        keys = ("total", "pde", "obs", "vm", "sp", "tb", "high", "vmax", "data", "bdata")
        self.history = {k: [] for k in keys}
        t0 = time.time()
        for ep in range(cfg.epochs):
            opt.zero_grad()
            L = _losses(self.sol, self.bnd, cfg, data)
            tot = _total(cfg, L)
            tot.backward()
            nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            self.history["total"].append(tot.item())
            for k in keys[1:]:
                self.history[k].append(L[k].item() if k in L else 0.0)
            if log_every and (ep % log_every == 0 or ep == cfg.epochs - 1):
                print(f"ep {ep:6d}  total {tot.item():.3e}  pde {L['pde'].item():.2e}  vm {L['vm'].item():.2e}"
                      f"  sp {L['sp'].item():.2e}  data {L['data'].item() if 'data' in L else 0.0:.2e}"
                      f"  bdata {L['bdata'].item() if 'bdata' in L else 0.0:.2e}  {time.time() - t0:.0f}s", flush=True)
            if ckpt and ep > 0 and ep % 1500 == 0:
                self.save(ckpt)
        self.sol.eval()
        self.bnd.eval()
        if ckpt:
            self.save(ckpt)
        return self

    # ---- inference
    def _cols(self, m, v, tau, params):
        m, v, tau = (torch.as_tensor(np.asarray(a, dtype=np.float32)).reshape(-1, 1) for a in (m, v, tau))
        n = max(m.shape[0], v.shape[0], tau.shape[0])
        m, v, tau = (a.expand(n, 1) if a.shape[0] == 1 else a for a in (m, v, tau))
        p = torch.as_tensor(np.asarray(params, dtype=np.float32))
        p = p.reshape(1, -1).expand(n, -1) if p.ndim == 1 or p.shape[0] == 1 else p
        return m, v, tau, p

    def boundary(self, v, tau, params) -> np.ndarray:
        _, v, tau, p = self._cols(np.ones(1), v, tau, params)
        with torch.no_grad():
            return self.bnd(v, tau, p).numpy().reshape(-1)

    def price_u(self, m, v, tau, params) -> np.ndarray:
        m, v, tau, p = self._cols(m, v, tau, params)
        with torch.no_grad():
            u = self.sol(torch.cat([m, v, tau, p], dim=1))
            b = self.bnd(v, tau, p)
        h = intrinsic(m)
        out = torch.where(m >= b, torch.maximum(u, h), h)
        return out.numpy().reshape(-1)

    def price(self, S, K, v, tau, params) -> np.ndarray:
        S, K = np.asarray(S, float), np.asarray(K, float)
        return K * self.price_u(S / K, v, tau, params)

    def delta(self, S, K, v, tau, params) -> np.ndarray:
        """du/dm: autograd on the solution network in the continuation region, -1 below the boundary."""
        m, v, tau, p = self._cols(np.asarray(S, float) / np.asarray(K, float), v, tau, params)
        mq = m.clone().requires_grad_(True)
        u = self.sol(torch.cat([mq, v, tau, p], dim=1))
        du = torch.autograd.grad(u, mq, grad_outputs=torch.ones_like(u))[0]
        with torch.no_grad():
            b = self.bnd(v, tau, p)
        return torch.where(m >= b, du, torch.full_like(du, -1.0)).detach().numpy().reshape(-1)

    # ---- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"sol": self.sol.state_dict(), "bnd": self.bnd.state_dict(), "kind": "coupled",
                    "config": json.dumps(asdict(self.config), default=list), "history": self.history}, path)

    @classmethod
    def load(cls, path: str | Path) -> "HestonCoupledPINN":
        blob = torch.load(path, weights_only=False)
        d = json.loads(blob["config"])
        box = ParamBox(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.pop("box").items()})
        obj = cls(CoupledConfig(box=box, **d))
        obj.sol.load_state_dict(blob["sol"])
        obj.bnd.load_state_dict(blob["bnd"])
        obj.sol.eval()
        obj.bnd.eval()
        obj.history = blob.get("history", {})
        return obj

    @property
    def net(self):  # lets code written for HestonCondPINN call .net.eval()
        return self.sol
