"""Training configuration for the European PINN.

`lambda_pde` is the one flag that turns this into either N0 or N1 from the
model ladder in `RESEARCH_GAP_AND_ROADMAP.md`:
  - lambda_pde=0.0  -> N0, the unstructured control. Only fit to the
    terminal payoff and the two boundary conditions; nothing constrains the
    interior, so it has no reason to match the BS PDE away from tau=0.
  - lambda_pde>0.0  -> N1, the physics-informed model. Same network, same
    terminal/boundary data, plus the PDE residual penalty everywhere in the
    interior.
Comparing N0 vs N1 interior error is the Phase 3 ablation: it's the direct,
checkable evidence that the physics loss is doing something, not just a
theoretical claim.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["EuropeanPINNConfig"]


@dataclass
class EuropeanPINNConfig:
    option_type: str = "put"

    # Black-Scholes parameters this instance is trained for.
    r: float = 0.03
    q: float = 0.0
    sigma: float = 0.2

    # Domain: moneyness m=S/K in [0, m_max], time-to-expiry tau in [0, tau_max].
    m_max: float = 3.0
    tau_max: float = 1.0

    # Network shape.
    hidden_layers: int = 4
    hidden_width: int = 64

    # Collocation point counts, resampled every epoch. Terminal points are
    # sampled half uniform, half concentrated near the payoff kink at m=1
    # (see `collocation.sample_terminal`) -- the kink is the hardest part of
    # the domain for a smooth network to fit, so it gets extra resolution.
    n_interior: int = 3000
    n_terminal: int = 400
    n_boundary: int = 200

    # Loss weights. See module docstring for what lambda_pde=0 means.
    # Terminal is weighted above 1 because it's what pins the PDE down to
    # *this* solution rather than some other function satisfying the PDE
    # and the two spatial boundary conditions.
    lambda_pde: float = 1.0
    lambda_terminal: float = 3.0
    lambda_boundary: float = 1.0

    lr: float = 2e-3
    lr_decay_every: int = 2000  # halve lr every this many epochs, 0 disables
    epochs: int = 6000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type!r}")
