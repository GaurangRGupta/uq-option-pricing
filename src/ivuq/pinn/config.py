"""Training configuration for the European (N0/N1) and American (N2) PINNs.

`lambda_pde` is the one flag that turns `EuropeanPINNConfig` into either N0
or N1 from the model ladder in `RESEARCH_GAP_AND_ROADMAP.md`:
  - lambda_pde=0.0  -> N0, the unstructured control. Only fit to the
    terminal payoff and the two boundary conditions; nothing constrains the
    interior, so it has no reason to match the BS PDE away from tau=0.
  - lambda_pde>0.0  -> N1, the physics-informed model. Same network, same
    terminal/boundary data, plus the PDE residual penalty everywhere in the
    interior.
Comparing N0 vs N1 interior error is the Phase 3 ablation: it's the direct,
checkable evidence that the physics loss is doing something, not just a
theoretical claim.

`AmericanPINNConfig.weighting_scheme` is N2's own ablation (Section 5/6 of
the roadmap: self-adaptive weighting is a training detail to fold into N2's
config, not a standalone phase) -- see `train_american.py` for what each of
"fixed", "curriculum", and "self_adaptive" actually does.

`HestonEuropeanPINNConfig` is Phase 3b's first step: the same N0/N1 idea
(`lambda_pde` switch) but under Heston's two-factor dynamics instead of
GBM -- validated against the closed-form `ivuq.pricing.heston.heston_price`
rather than Black-Scholes. This is deliberately built and checked against a
known solution *before* attempting the American-Heston free-boundary PINN,
which has no closed form to validate against at all.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["EuropeanPINNConfig", "AmericanPINNConfig", "HestonEuropeanPINNConfig"]


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


@dataclass
class AmericanPINNConfig:
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

    # Collocation point counts. Under "fixed"/"curriculum" these are
    # resampled fresh every epoch, same as the European PINN. Under
    # "self_adaptive" the interior points are sampled once and held fixed
    # for the whole run -- see train_american.py for why.
    n_interior: int = 3000
    n_terminal: int = 400
    n_boundary: int = 200

    # Fixed loss weights for the two data-fit terms (terminal payoff, the
    # two spatial boundaries). These stay plain scalars under every
    # weighting scheme -- they live on lower-dimensional slices with no
    # free boundary to resolve, so there's nothing for spatial adaptivity
    # to buy here.
    lambda_terminal: float = 3.0
    lambda_boundary: float = 1.0

    # Fixed loss weights for the three LCP penalty terms (free_boundary.py
    # equation (3)): ineq_violation, obstacle_violation, complementarity_violation.
    # Under "fixed" these are used as-is. Under "curriculum" they're the
    # *target* weights the ramp climbs to. Under "self_adaptive" they're
    # ignored for the interior terms -- per-point trainable weights replace
    # them (see train_american.py).
    lambda_ineq: float = 1.0
    lambda_obstacle: float = 1.0
    lambda_complementarity: float = 1.0

    # Which of the three loss-weighting strategies to train with -- the
    # Phase 3 ablation. "fixed": constant weights above, fresh collocation
    # every epoch (same recipe as the European PINN). "curriculum": weights
    # ramp up linearly over the first `curriculum_ramp_epochs`, plus
    # resampling shifts to concentrate points near the current estimated
    # exercise boundary once the ramp is under way. "self_adaptive": a
    # simplified, points-only version of McClenny & Braga-Neto
    # (arXiv:2009.04544) -- trainable per-point weights on the three LCP
    # terms, updated by gradient ascent so the network can't just ignore
    # persistently-hard points (e.g. near the free boundary).
    weighting_scheme: str = "fixed"

    # "curriculum" only: ramp lambda_ineq/obstacle/complementarity linearly
    # from curriculum_start_frac*target to target over this many epochs.
    curriculum_ramp_epochs: int = 1500
    curriculum_start_frac: float = 0.1

    lr: float = 2e-3
    lr_decay_every: int = 2000  # halve lr every this many epochs, 0 disables
    epochs: int = 6000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type!r}")
        if self.weighting_scheme not in ("fixed", "curriculum", "self_adaptive"):
            raise ValueError(
                "weighting_scheme must be 'fixed', 'curriculum', or 'self_adaptive', "
                f"got {self.weighting_scheme!r}"
            )


@dataclass
class HestonEuropeanPINNConfig:
    option_type: str = "put"

    # Rates and Heston's own five parameters.
    r: float = 0.03
    q: float = 0.0
    kappa: float = 2.0     # mean-reversion speed of variance
    theta: float = 0.04    # long-run variance
    xi: float = 0.4        # vol-of-vol
    rho: float = -0.5      # correlation between the two Brownian motions
    v0: float = 0.04       # initial variance -- used at inference (.price()), not sampled in training

    # Domain: moneyness m=S/K in [0, m_max], variance v in [0, v_max],
    # time-to-expiry tau in [0, tau_max]. v_max needs to comfortably cover
    # where theta/v0 and their fluctuations under kappa/xi actually live --
    # 4x theta is a generous default, not a tuned one.
    m_max: float = 3.0
    v_max: float = 0.16
    tau_max: float = 1.0

    # Network shape. One more input dimension than EuropeanPINNConfig (m, v,
    # tau instead of m, tau); see network.PINN's in_dim parameter.
    hidden_layers: int = 4
    hidden_width: int = 64

    # Collocation point counts, resampled every epoch.
    n_interior: int = 4000
    n_terminal: int = 500
    n_boundary_m: int = 200
    n_boundary_v: int = 200

    # Loss weights. lambda_pde=0 gives N0-Heston (the control); >0 gives
    # N1-Heston. Same rationale as EuropeanPINNConfig for why terminal is
    # weighted above 1.
    lambda_pde: float = 1.0
    lambda_terminal: float = 3.0
    lambda_boundary_m: float = 1.0
    lambda_boundary_v: float = 1.0

    lr: float = 2e-3
    lr_decay_every: int = 2000
    epochs: int = 6000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type!r}")
