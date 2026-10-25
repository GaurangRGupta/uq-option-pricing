"""Parameter-conditioned Heston American-put PINN.

Why. The earlier Heston PINNs (`american_heston.py`, `european_heston.py`) are
trained for one fixed parameter set, so a different set needs a new training
run. Fitting Heston to a day of market quotes needs the price at hundreds of
parameter sets, and again every day. Here the five-parameter family is an
input of the network, so one training run per underlying prices the whole
family and is differentiable in the parameters (calibration by autograd).

Equation. With m = S/K, u = V/K, tau = T - t, and p = (kappa, theta, xi, rho, r, q):

    R[u] = u_tau - 0.5 v m^2 u_mm - rho xi v m u_mv - 0.5 xi^2 v u_vv
           - (r - q) m u_m - kappa (theta - v) u_v + r u                  (1)

(the same operator as `heston_pde.pde_residual`, eq. (4) there). The American
put is the linear complementarity problem with h(m) = max(1 - m, 0):

    R[u] >= 0,    u >= h,    R[u] (u - h) = 0   <=>   min(R[u], u - h) = 0   (2)

Loss (all terms are mean squares over points resampled every epoch, with the
parameters p drawn uniformly from the box at every point):

    L = w_lcp  * E[ min(R[u], u - h)^2 ]                   interior, eq. (2)
      + w_term * E[ (u(m, v, 0, p) - h(m))^2 ]              terminal payoff
      + w_low  * E[ (u(m_lo, v, tau, p) - (1 - m_lo))^2 ]   deep ITM: exercised
      + w_high * E[ u(m_hi, v, tau, p)^2 ]                  deep OTM: worthless
      + w_vmax * E[ u_vv(m, v_max, tau, p)^2 ]              far-variance condition
      + w_data * E[ (u - u_FD)^2 ]                          optional anchor (default 0)

The last term is off by default: with w_data = 0 the network is a pure PINN
and sees no prices. When it is on (see `HestonCondConfig.w_data`) the labels
come from `pricing/heston_fd.py`, and the model is reported as the anchored
variant, never as a pure PINN.

Domain. m in [m_lo, m_hi], v in [0, v_max], tau in [0, tau_max]. The data this
is used on has tau up to 70 days and K/S between 0.8 and 1.25, so tau_max =
0.30 and m in [0.3, 2.0] leave margin for the two Dirichlet edges to be good
approximations (the same truncation the FD reference uses).

Inputs are rescaled to [-1, 1] over the box before the first layer; the
derivatives in (1) are taken with respect to the physical m, v, tau.

Control variate (`control_variate=True`). A plain network has to learn the whole
surface, including the payoff kink at m = 1, in nine dimensions; pilot runs
(DEVLOG 2026-10-23) were off by $0.67 per $100 strike. With the flag on, the
network represents only the difference from a known price:

    u(m, v, tau, p) = BS(m, sigma^2 = v, tau; r, q) + (tau / tau_max) N(m, v, tau, p)     (3)

BS is the European Black-Scholes put per unit strike at variance v,

    BS = e^{-r tau} Phi(-d2) - m e^{-q tau} Phi(-d1),  d1 = (ln m + (r - q + v/2) tau) / sqrt(v tau),
    d2 = d1 - sqrt(v tau),

which is smooth for tau > 0 and equals the payoff at tau = 0. So (3) satisfies the
terminal condition exactly (the factor tau kills N there), and N only carries the
Heston-versus-constant-variance difference and the early-exercise premium, both small
and smooth. The PDE residual (1) and the loss are unchanged and are taken on u in (3);
autograd differentiates through BS. v is floored at 1e-6 and tau at 1e-7 inside BS only.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

__all__ = ["ParamBox", "HestonCondConfig", "CondNet", "HestonCondPINN", "bs_put_unit", "lcp_terms", "sample_params"]

PARAM_NAMES = ("kappa", "theta", "xi", "rho", "r", "q")


@dataclass
class ParamBox:
    """Axis-aligned box for (kappa, theta, xi, rho, r, q) plus the state domain."""

    kappa: tuple = (0.5, 6.0)
    theta: tuple = (0.01, 0.09)
    xi: tuple = (0.15, 1.2)
    rho: tuple = (-0.95, 0.0)
    r: tuple = (0.03, 0.05)
    q: tuple = (0.0, 0.02)
    m_lo: float = 0.3
    m_hi: float = 2.0
    v_max: float = 0.30
    tau_max: float = 0.30

    def lo_hi(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([getattr(self, n)[0] for n in PARAM_NAMES])
        hi = np.array([getattr(self, n)[1] for n in PARAM_NAMES])
        return lo, hi


@dataclass
class HestonCondConfig:
    box: ParamBox = field(default_factory=ParamBox)
    hidden_layers: int = 5
    hidden_width: int = 128
    control_variate: bool = False   # eq. (3) in the module docstring
    n_interior: int = 4000
    n_terminal: int = 600
    n_edge: int = 300
    n_data: int = 2000
    w_lcp: float = 1.0
    w_term: float = 5.0
    w_low: float = 1.0
    w_high: float = 1.0
    w_vmax: float = 0.1
    w_data: float = 0.0
    lr: float = 2e-3
    lr_final: float = 5e-5
    epochs: int = 6000
    seed: int = 0


class CondNet(nn.Module):
    """tanh MLP on the 9 inputs (m, v, tau, kappa, theta, xi, rho, r, q), rescaled to [-1, 1]."""

    def __init__(self, box: ParamBox, hidden_layers: int, hidden_width: int, control_variate: bool = False) -> None:
        super().__init__()
        self.control_variate = control_variate
        self.tau_max = box.tau_max
        plo, phi = box.lo_hi()
        lo = np.concatenate([[box.m_lo, 0.0, 0.0], plo])
        hi = np.concatenate([[box.m_hi, box.v_max, box.tau_max], phi])
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(hi, dtype=torch.float32))
        layers: list[nn.Module] = []
        d = 9
        for _ in range(hidden_layers):
            layers += [nn.Linear(d, hidden_width), nn.Tanh()]
            d = hidden_width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0
        out = self.net(z)
        if not self.control_variate:
            return out
        m, v, tau = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        r, q = x[:, 7:8], x[:, 8:9]
        return bs_put_unit(m, v, tau, r, q) + (tau / self.tau_max) * out


def bs_put_unit(m, var, tau, r, q):
    """European Black-Scholes put per unit strike at variance `var` (eq. (3) in the module docstring)."""
    var = torch.clamp(var, min=1e-6)
    t = tau + 1e-7
    sd = torch.sqrt(var * t)
    d1 = (torch.log(m) + (r - q + 0.5 * var) * t) / sd
    d2 = d1 - sd
    return torch.exp(-r * t) * torch.special.ndtr(-d2) - m * torch.exp(-q * t) * torch.special.ndtr(-d1)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def sample_params(n: int, box: ParamBox, gen: torch.Generator | None = None) -> torch.Tensor:
    lo, hi = box.lo_hi()
    lo_t, hi_t = torch.tensor(lo, dtype=torch.float32), torch.tensor(hi, dtype=torch.float32)
    return lo_t + (hi_t - lo_t) * torch.rand(n, len(PARAM_NAMES), generator=gen)


def _sample_m(n: int, box: ParamBox) -> torch.Tensor:
    """70% of points in the band the data lives in, 30% over the full domain."""
    n_band = int(0.7 * n)
    band = 0.7 + 0.7 * torch.rand(n_band, 1)
    full = box.m_lo + (box.m_hi - box.m_lo) * torch.rand(n - n_band, 1)
    return torch.cat([band, full], dim=0)


def _sample_v(p: torch.Tensor, box: ParamBox) -> torch.Tensor:
    """Variance concentrated around the long-run level theta (ratio v/theta between 0.1 and 3),
    plus a uniform share over [0, v_max] so the whole box is covered."""
    n = p.shape[0]
    theta = p[:, 1:2]
    ratio = torch.exp(np.log(0.1) + (np.log(3.0) - np.log(0.1)) * torch.rand(n, 1))
    v_rel = (theta * ratio).clamp(max=box.v_max)
    v_uni = box.v_max * torch.rand(n, 1)
    pick = (torch.rand(n, 1) < 0.7).float()
    return pick * v_rel + (1 - pick) * v_uni


def _sample_tau(n: int, box: ParamBox) -> torch.Tensor:
    return box.tau_max * torch.rand(n, 1) ** 1.3


def intrinsic(m: torch.Tensor) -> torch.Tensor:
    return torch.clamp(1.0 - m, min=0.0)


# ---------------------------------------------------------------------------
# PDE terms
# ---------------------------------------------------------------------------

def lcp_terms(net: nn.Module, m, v, tau, p):
    """Returns (R, u): the PDE operator (1) and the network value at the points."""
    m = m.clone().requires_grad_(True)
    v = v.clone().requires_grad_(True)
    tau = tau.clone().requires_grad_(True)
    u = net(torch.cat([m, v, tau, p], dim=1))
    ones = torch.ones_like(u)
    du_dm, du_dv, du_dtau = torch.autograd.grad(u, (m, v, tau), grad_outputs=ones, create_graph=True)
    d2m = torch.autograd.grad(du_dm, m, grad_outputs=torch.ones_like(du_dm), create_graph=True)[0]
    d2v = torch.autograd.grad(du_dv, v, grad_outputs=torch.ones_like(du_dv), create_graph=True)[0]
    dmv = torch.autograd.grad(du_dm, v, grad_outputs=torch.ones_like(du_dm), create_graph=True)[0]
    kappa, theta, xi, rho, r, q = (p[:, i : i + 1] for i in range(6))
    R = (
        du_dtau
        - 0.5 * v * m**2 * d2m
        - rho * xi * v * m * dmv
        - 0.5 * xi**2 * v * d2v
        - (r - q) * m * du_dm
        - kappa * (theta - v) * du_dv
        + r * u
    )
    return R, u


def _vmax_term(net, m, tau, p, v_max):
    m = m.clone().requires_grad_(True)
    v = torch.full_like(m, v_max).requires_grad_(True)
    u = net(torch.cat([m, v, tau, p], dim=1))
    du_dv = torch.autograd.grad(u, v, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    return torch.autograd.grad(du_dv, v, grad_outputs=torch.ones_like(du_dv), create_graph=True)[0]


def _losses(net: CondNet, cfg: HestonCondConfig, data=None) -> dict[str, torch.Tensor]:
    box = cfg.box
    n = cfg.n_interior
    p = sample_params(n, box)
    m, v, tau = _sample_m(n, box), _sample_v(p, box), _sample_tau(n, box)
    R, u = lcp_terms(net, m, v, tau, p)
    loss_lcp = torch.mean(torch.minimum(R, u - intrinsic(m)) ** 2)

    nt = cfg.n_terminal
    pt = sample_params(nt, box)
    mt = torch.cat([_sample_m(nt // 2, box), (1.0 + 0.1 * torch.randn(nt - nt // 2, 1)).clamp(box.m_lo, box.m_hi)], dim=0)
    vt, taut = _sample_v(pt, box), torch.zeros(nt, 1)
    loss_term = torch.mean((net(torch.cat([mt, vt, taut, pt], dim=1)) - intrinsic(mt)) ** 2)

    ne = cfg.n_edge
    pe = sample_params(ne, box)
    ve, taue = _sample_v(pe, box), _sample_tau(ne, box)
    ulo = net(torch.cat([torch.full((ne, 1), box.m_lo), ve, taue, pe], dim=1))
    uhi = net(torch.cat([torch.full((ne, 1), box.m_hi), ve, taue, pe], dim=1))
    loss_low = torch.mean((ulo - (1.0 - box.m_lo)) ** 2)
    loss_high = torch.mean(uhi**2)

    pv = sample_params(ne, box)
    d2v = _vmax_term(net, _sample_m(ne, box), _sample_tau(ne, box), pv, box.v_max)
    loss_vmax = torch.mean(d2v**2)

    out = {"lcp": loss_lcp, "term": loss_term, "low": loss_low, "high": loss_high, "vmax": loss_vmax}
    if cfg.w_data > 0 and data is not None:
        X, y = data
        idx = torch.randint(0, X.shape[0], (cfg.n_data,))
        out["data"] = torch.mean((net(X[idx]) - y[idx]) ** 2)
    return out


def _total(cfg: HestonCondConfig, L: dict[str, torch.Tensor]) -> torch.Tensor:
    tot = cfg.w_lcp * L["lcp"] + cfg.w_term * L["term"] + cfg.w_low * L["low"] + cfg.w_high * L["high"] + cfg.w_vmax * L["vmax"]
    if "data" in L:
        tot = tot + cfg.w_data * L["data"]
    return tot


# ---------------------------------------------------------------------------
# wrapper
# ---------------------------------------------------------------------------

class HestonCondPINN:
    """Trained network plus its config. Prices in u = V/K units or dollars."""

    def __init__(self, config: HestonCondConfig) -> None:
        self.config = config
        self.net = CondNet(config.box, config.hidden_layers, config.hidden_width, config.control_variate)
        self.history: dict[str, list[float]] = {}

    # ---- training
    def fit(self, data: tuple[torch.Tensor, torch.Tensor] | None = None, log_every: int = 250, ckpt: str | Path | None = None):
        cfg = self.config
        torch.manual_seed(cfg.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        gamma = (cfg.lr_final / cfg.lr) ** (1.0 / max(cfg.epochs, 1))
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)
        self.history = {k: [] for k in ("total", "lcp", "term", "low", "high", "vmax", "data")}
        t0 = time.time()
        for ep in range(cfg.epochs):
            opt.zero_grad()
            L = _losses(self.net, cfg, data)
            tot = _total(cfg, L)
            tot.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
            opt.step()
            sched.step()
            self.history["total"].append(tot.item())
            for k in ("lcp", "term", "low", "high", "vmax", "data"):
                self.history[k].append(L[k].item() if k in L else 0.0)
            if log_every and (ep % log_every == 0 or ep == cfg.epochs - 1):
                print(f"ep {ep:6d}  total {tot.item():.3e}  lcp {L['lcp'].item():.2e}  term {L['term'].item():.2e}"
                      f"  data {L['data'].item() if 'data' in L else 0.0:.2e}  {time.time() - t0:.0f}s", flush=True)
            if ckpt and ep > 0 and ep % 2000 == 0:
                self.save(ckpt)
        self.net.eval()
        if ckpt:
            self.save(ckpt)
        return self

    # ---- inference
    def _x(self, m, v, tau, params) -> torch.Tensor:
        m, v, tau = (torch.as_tensor(np.asarray(a, dtype=np.float32)).reshape(-1, 1) for a in (m, v, tau))
        p = torch.as_tensor(np.asarray(params, dtype=np.float32))
        p = p.reshape(1, -1).expand(m.shape[0], -1) if p.ndim == 1 or p.shape[0] == 1 else p
        return torch.cat([m, v, tau, p], dim=1)

    def price_u(self, m, v, tau, params) -> np.ndarray:
        """u = V/K at (m, v, tau) for parameter vector(s) params = (kappa, theta, xi, rho, r, q).
        Never below the payoff: the exact price satisfies u >= h, so max(u, h) can only reduce error."""
        x = self._x(m, v, tau, params)
        with torch.no_grad():
            u = self.net(x).numpy().reshape(-1)
        return np.maximum(u, np.maximum(1.0 - x[:, 0].numpy(), 0.0))

    def price(self, S, K, v, tau, params) -> np.ndarray:
        S, K = np.asarray(S, float), np.asarray(K, float)
        return K * self.price_u(S / K, v, tau, params)

    def delta(self, S, K, v, tau, params) -> np.ndarray:
        """dV/dS = du/dm by autograd on the network (before the max with the payoff)."""
        x = self._x(np.asarray(S, float) / np.asarray(K, float), v, tau, params).clone()
        x[:, 0].requires_grad_(True)
        m = x[:, 0:1].clone().requires_grad_(True)
        u = self.net(torch.cat([m, x[:, 1:]], dim=1))
        du = torch.autograd.grad(u, m, grad_outputs=torch.ones_like(u))[0]
        return du.detach().numpy().reshape(-1)

    # ---- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state": self.net.state_dict(), "config": json.dumps(asdict(self.config), default=list),
                    "history": self.history}, path)

    @classmethod
    def load(cls, path: str | Path) -> "HestonCondPINN":
        blob = torch.load(path, weights_only=False)
        d = json.loads(blob["config"])
        box = ParamBox(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in d.pop("box").items()})
        obj = cls(HestonCondConfig(box=box, **d))
        obj.net.load_state_dict(blob["state"])
        obj.net.eval()
        obj.history = blob.get("history", {})
        return obj
