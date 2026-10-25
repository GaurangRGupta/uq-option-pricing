"""Heston parameter boxes per underlying, matched to the DoltHub window 2025-10 to 2026-10
(ATM implied variance: SPY 0.01 to 0.06, AAPL 0.03 to 0.12, NVDA 0.08 to 0.32; trailing dividend
yield SPY 1.0%, AAPL 0.36%, NVDA 0.08%; r between 3.6% and 4.1%). Used by training, calibration
and the FD reference so all three describe the same domain."""

from .heston_cond import ParamBox

__all__ = ["BOXES", "box_for"]

BOXES = {
    "SPY": dict(theta=(0.01, 0.09), v_max=0.30, q=(0.009, 0.015)),
    "AAPL": dict(theta=(0.02, 0.16), v_max=0.55, q=(0.002, 0.006)),
    "NVDA": dict(theta=(0.06, 0.40), v_max=0.90, q=(0.0, 0.003)),
}


def box_for(symbol: str, r=(0.034, 0.043), wide: bool = False) -> ParamBox:
    """wide=True is for FD calibration only (the PINNs are trained on the narrow box and must not be
    queried outside it): looser kappa, xi, rho and a higher long-run variance cap."""
    cfg = dict(BOXES[symbol])
    if wide:
        lo, hi = cfg["theta"]
        cfg["theta"] = (lo, 1.6 * hi)
        cfg["v_max"] = max(cfg["v_max"], 1.6 * hi * 2)
        return ParamBox(r=r, kappa=(0.3, 12.0), xi=(0.10, 2.5), rho=(-0.99, 0.2), **cfg)
    return ParamBox(r=r, **cfg)
