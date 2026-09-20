"""Deep-learning baselines, from plain to structure-aware.

All are trained by mean squared error on labelled prices u = V/K over inputs
(m, tau), with the same training loop (`TorchRegressor`): Adam, step decay,
15% of labels held out for early stopping. None has a PDE term.

  mlp_tanh / mlp_relu : plain feed-forward nets. mlp_tanh matches the PINN's
      architecture exactly (4 x 64, tanh), so it isolates the effect of the
      PDE loss and of labels.

  resnet_silu : residual MLP with SiLU activations. Skip connections and a
      smooth non-saturating activation are the standard modern default for
      tabular regression.

  fourier_mlp : the inputs are lifted to random Fourier features
      [x, sin(2 pi B x), cos(2 pi B x)] before the network. Plain MLPs learn low
      frequencies first (spectral bias) and smear the payoff kink at m = 1;
      Fourier features are the standard remedy (Tancik et al. 2020).

  picnn : a partially input-convex network (Amos, Xu, Kolter 2017). The
      output is convex in moneyness m by construction, for every tau, so it
      cannot violate the no-arbitrage convexity condition d2u/dm2 >= 0. It
      follows the "shape-constrained neural network" approach used for
      arbitrage-free surface fitting. Structure: with y = m the convex
      argument and u_0 = tau the context,
        u_{i+1} = act(W_i u_i + b_i),
        z_{i+1} = g( Wz_i (z_i * relu(Az_i u_i + c_i)) + Wy_i (y * (Ay_i u_i + d_i)) + Bu_i u_i + e_i ),
      with Wz_i >= 0 (softplus-parameterised), g = softplus (convex, nondecreasing),
      and g = identity on the final layer. Every term is convex in y, and
      nonnegative combinations and convex nondecreasing compositions of convex
      functions are convex.

  bs_iv_net : the price is the Black-Scholes formula evaluated at an
      implied volatility sigma(m, tau) = 0.01 + softplus(net(m, tau)) that the
      network learns, u = BS(m, 1, tau, r, q, sigma(m, tau)). This is the
      "deep implied volatility" hybrid: domain knowledge enters through the
      closed form, but no PDE is enforced on the learned surface. It is
      exactly right for European options and structurally misspecified for
      American ones (a European price cannot reach the early-exercise premium),
      which is the point of including it.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ivuq.baselines.data import MarketParams

__all__ = ["TorchRegressor", "MLP", "ResNetMLP", "FourierMLP", "PICNN", "BSIVNet"]


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, layers: int = 4, width: int = 64, activation: str = "tanh") -> None:
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        mods: list[nn.Module] = []
        d = in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, width), act()]
            d = width
        mods.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.f1, self.f2 = nn.Linear(width, width), nn.Linear(width, width)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.f2(F.silu(self.f1(h)))


class ResNetMLP(nn.Module):
    def __init__(self, in_dim: int = 2, blocks: int = 3, width: int = 64) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.Sequential(*[_ResBlock(width) for _ in range(blocks)])
        self.out = nn.Linear(width, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.silu(self.blocks(F.silu(self.inp(x)))))


class FourierMLP(nn.Module):
    def __init__(self, in_dim: int = 2, n_freq: int = 48, scale: float = 3.0, layers: int = 3, width: int = 64) -> None:
        super().__init__()
        self.register_buffer("B", torch.randn(in_dim, n_freq) * scale)
        self.mlp = MLP(in_dim + 2 * n_freq, layers, width, "silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * x @ self.B
        return self.mlp(torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=1))


class PICNN(nn.Module):
    """Convex in x[:, 0] (moneyness) for every value of x[:, 1] (maturity)."""

    def __init__(self, width: int = 48, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth
        u_dims = [1] + [width] * depth
        z_dims = [width] * depth + [1]
        self.u_next = nn.ModuleList([nn.Linear(u_dims[i], u_dims[i + 1]) for i in range(depth)])
        self.wu = nn.ModuleList([nn.Linear(u_dims[i], z_dims[i]) for i in range(depth + 1)])
        self.gy = nn.ModuleList([nn.Linear(u_dims[i], 1) for i in range(depth + 1)])
        self.wy = nn.ModuleList([nn.Linear(1, z_dims[i], bias=False) for i in range(depth + 1)])
        self.gz = nn.ModuleList([nn.Linear(u_dims[i], z_dims[i - 1]) for i in range(1, depth + 1)])
        self.wz_raw = nn.ParameterList(
            [nn.Parameter(torch.randn(z_dims[i], z_dims[i - 1]) * 0.1 - 1.0) for i in range(1, depth + 1)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, u = x[:, 0:1], x[:, 1:2]
        z = None
        for i in range(self.depth + 1):
            term = self.wu[i](u) + self.wy[i](y * self.gy[i](u))
            if i >= 1:
                gated = z * F.relu(self.gz[i - 1](u))
                term = term + F.linear(gated, F.softplus(self.wz_raw[i - 1]))
            z = F.softplus(term) if i < self.depth else term
            if i < self.depth:
                u = torch.tanh(self.u_next[i](u))
        return z


class BSIVNet(nn.Module):
    def __init__(self, option_type: str, params: MarketParams, layers: int = 3, width: int = 64) -> None:
        super().__init__()
        self.is_call = option_type == "call"
        self.r, self.q = params.r, params.q
        self.mlp = MLP(2, layers, width, "tanh")
        # Start near sigma = 0.2: softplus(-1.65) + 0.01 ~ 0.2.
        nn.init.constant_(self.mlp.net[-1].bias, -1.65)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m, tau = x[:, 0:1].clamp_min(1e-6), x[:, 1:2].clamp_min(1e-6)
        sigma = 0.01 + F.softplus(self.mlp(x))
        vol_t = sigma * torch.sqrt(tau)
        d1 = (torch.log(m) + (self.r - self.q + 0.5 * sigma**2) * tau) / vol_t
        d2 = d1 - vol_t

        def cdf(z: torch.Tensor) -> torch.Tensor:
            return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

        disc_r, disc_q = torch.exp(-self.r * tau), torch.exp(-self.q * tau)
        if self.is_call:
            return m * disc_q * cdf(d1) - disc_r * cdf(d2)
        return disc_r * cdf(-d2) - m * disc_q * cdf(-d1)


class TorchRegressor:
    """Shared training loop. `build(in_dim)` returns a fresh nn.Module."""

    def __init__(
        self,
        name: str,
        build,
        epochs: int = 3000,
        lr: float = 2e-3,
        lr_decay_every: int = 1000,
        val_frac: float = 0.15,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.build = build
        self.epochs, self.lr, self.lr_decay_every = epochs, lr, lr_decay_every
        self.val_frac, self.seed = val_frac, seed
        self.net: nn.Module | None = None

    def fit(self, X: np.ndarray, u: np.ndarray) -> "TorchRegressor":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n = len(u)
        perm = rng.permutation(n)
        n_val = int(round(self.val_frac * n))
        val_idx, tr_idx = (perm[:n_val], perm[n_val:]) if n_val >= 2 else (None, perm)

        Xt = torch.tensor(X, dtype=torch.float32)
        ut = torch.tensor(u, dtype=torch.float32).reshape(-1, 1)
        X_tr, u_tr = Xt[tr_idx], ut[tr_idx]
        X_va, u_va = (Xt[val_idx], ut[val_idx]) if val_idx is not None else (None, None)

        net = self.build(X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        sched = (
            torch.optim.lr_scheduler.StepLR(opt, step_size=self.lr_decay_every, gamma=0.5)
            if self.lr_decay_every > 0
            else None
        )
        best, best_state = float("inf"), copy.deepcopy(net.state_dict())
        for epoch in range(self.epochs):
            opt.zero_grad()
            loss = torch.mean((net(X_tr) - u_tr) ** 2)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if epoch % 25 == 0 or epoch == self.epochs - 1:
                with torch.no_grad():
                    score = float(torch.mean((net(X_va) - u_va) ** 2)) if X_va is not None else float(loss)
                if score < best:
                    best, best_state = score, copy.deepcopy(net.state_dict())
        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            out = self.net(torch.tensor(X, dtype=torch.float32)).numpy()
        return out.astype(np.float64).ravel()
