"""The network architecture: a plain feed-forward net over (m, tau).

Tanh activations, not ReLU -- the PDE residual needs a second derivative in
m, and ReLU's second derivative is zero almost everywhere, which kills the
PDE loss's gradient. Standard PINN choice, not a stylistic one.

(An earlier version of this file tried to hard-enforce the terminal
condition via an ansatz u = payoff(m) + tau*correction(m,tau). That's a
real technique, but it back-fires here: the payoff has a kink at m=1, and
adding it back in for every tau (not just tau=0) forces `correction` to
cancel that same kink at every tau, which a smooth tanh network can't do
exactly -- it produced a visible error band around m=1 for all tau, worse
than just penalizing the terminal mismatch directly. Soft-constraining the
terminal condition via a loss term (see `train.py`), same as the papers in
`planning/papers/PAPER_TRAIL.md`, is the simpler and more accurate choice
here.)
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["PINN"]


class PINN(nn.Module):
    def __init__(self, hidden_layers: int = 4, hidden_width: int = 32, in_dim: int = 2) -> None:
        """`in_dim` is 2 for (m, tau) under GBM, 3 for (m, v, tau) under
        Heston -- same architecture either way, just one more input."""
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_width))
            layers.append(nn.Tanh())
            in_dim = hidden_width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
