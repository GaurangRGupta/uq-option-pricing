"""Daily walk-forward theo from a Heston pricer.

For each underlying, dates are processed in order and a date only ever uses
quotes from itself and earlier dates.

Mode A, "full": on every date fit (kappa, theta, xi, rho, v0) to that date's quotes,
with a soft pull toward the previous date's fit (Tikhonov weight `prior_weight`
on the box-scaled parameters), then theo_i = K_i u(m_i, v0, tau_i; p). The
residual mid - theo is what the fit could not explain with five parameters.

Mode B, "slow": (kappa, theta, xi, rho) are estimated jointly with one v0 per
day from the previous `window` dates only; on date t only v0_t is fitted, to
that date's quotes, with the slow parameters held fixed. The surface can then
not bend to absorb a quote it disagrees with, apart from its overall variance
level. Dates before `min_hist` earlier dates exist use mode A and are flagged
(`burn_in`).

Output columns added to the quotes: theo, model_delta (= du/dm = dV/dS), kappa, theta,
xi, rho, v0, fit_rmse (RMS of residual / c_i on that date), burn_in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ivuq.calibration.heston_calib import GLOBAL, calibrate

__all__ = ["atm_variance", "run_theo"]


def atm_variance(g: pd.DataFrame) -> float:
    near = g[(g["m"] - 1.0).abs() < 0.05]
    near = near if len(near) >= 3 else g
    short = near[near["dte"] <= near["dte"].quantile(0.5)]
    return float(np.median((short if len(short) else near)["vendor_iv"]) ** 2)


def _start(box, g) -> tuple[dict, float]:
    va = float(np.clip(atm_variance(g), 0.004, 0.9 * box.v_max))
    th = float(np.clip(va, box.theta[0] * 1.05, box.theta[1] * 0.95))
    return dict(kappa=2.0, theta=th, xi=0.5, rho=-0.6), va


def _price(pricer, g, params, v0):
    n = len(g)
    P = np.column_stack([np.full(n, params["kappa"]), np.full(n, params["theta"]), np.full(n, params["xi"]),
                         np.full(n, params["rho"]), g["r"].to_numpy(float), g["q"].to_numpy(float)])
    m, tau, K = g["m"].to_numpy(float), g["tau"].to_numpy(float), g["strike"].to_numpy(float)
    v = np.full(n, v0)
    return K * pricer.price_u(m, v, tau, P), pricer.delta_u(m, v, tau, P)


def _fit_full(pricer, g, prev, prior_weight, scale):
    box = pricer.box
    p0, va = _start(box, g)
    starts = [(p0, va)]
    if prev is not None:
        starts.append((prev["params"], prev["v0"]))
    best = None
    for pp, vv in starts:
        r = calibrate(pricer, g, pp, vv, free=GLOBAL + ("v0",), prior=None if prev is None else prev["params"],
                      prior_weight=0.0 if prev is None else prior_weight, scale=scale)
        if best is None or r.cost < best.cost:
            best = r
    return best


def run_theo(pricer, df: pd.DataFrame, mode: str = "full", prior_weight: float = 0.3, window: int = 8,
             min_hist: int = 3, max_nfev: int = 40, verbose: bool = False, final_pricer=None,
             warmup_dates: int = 0, refit_every: int = 1) -> pd.DataFrame:
    """df: one underlying, all dates, with a `scale` column. Returns df with the theo columns.

    final_pricer: if given, theo and delta are re-evaluated with it at the fitted parameters (the
    fit itself still uses `pricer`), e.g. a fine-grid FD solve after a cheaper calibration.
    warmup_dates: the first dates only warm up the parameter history; their rows are dropped.
    refit_every: slow mode re-estimates (kappa, theta, xi, rho) every this many dates (past dates only in
    between it keeps the last estimate); v0 is always refitted on the current date."""
    out = df.copy()
    for c in ("theo", "model_delta", "kappa", "theta", "xi", "rho", "v0", "fit_rmse"):
        out[c] = np.nan
    out["burn_in"] = False
    dates = sorted(out["date"].unique())
    by_date = {d: out.index[out["date"] == d] for d in dates}
    prev, slow_prev = None, None
    for k, d in enumerate(dates):
        idx = by_date[d]
        g = out.loc[idx]
        scale = g["scale"].to_numpy()
        use_slow = mode == "slow" and k >= min_hist
        if not use_slow:
            res = _fit_full(pricer, g, prev, prior_weight, scale)
            params, v0 = res.params, float(res.v0[0])
            out.loc[idx, "burn_in"] = mode == "slow"
        else:
            if slow_prev is None or k % refit_every == 0:
                past = dates[max(0, k - window):k]
                panel = out.loc[np.concatenate([by_date[x] for x in past])]
                day_idx = np.concatenate([np.full(len(by_date[x]), j) for j, x in enumerate(past)])
                v0_init = np.array([atm_variance(out.loc[by_date[x]]) for x in past])
                v0_init = np.clip(v0_init, 0.004, 0.9 * pricer.box.v_max)
                base = slow_prev if slow_prev is not None else _start(pricer.box, g)[0]
                rp = calibrate(pricer, panel, base, v0_init, day_idx=day_idx, free=GLOBAL + ("v0",),
                               prior=slow_prev, prior_weight=0.0 if slow_prev is None else prior_weight,
                               max_nfev=max_nfev, scale=panel["scale"].to_numpy())
                slow_prev = rp.params
            va = float(np.clip(atm_variance(g), 0.004, 0.9 * pricer.box.v_max))
            res = calibrate(pricer, g, slow_prev, va, free=("v0",), scale=scale)
            params, v0 = slow_prev, float(res.v0[0])
        theo, delta = _price(final_pricer or pricer, g, params, v0)
        out.loc[idx, "theo"] = theo
        out.loc[idx, "model_delta"] = delta
        for kname in GLOBAL:
            out.loc[idx, kname] = params[kname]
        out.loc[idx, "v0"] = v0
        out.loc[idx, "fit_rmse"] = float(np.sqrt(np.mean(((theo - g["mid"].to_numpy()) / scale) ** 2)))
        if not use_slow:
            prev = {"params": params, "v0": v0}
        if verbose and k % 10 == 0:
            print(f"{d} {mode} rmse={out.loc[idx, 'fit_rmse'].iloc[0]:.2f} v0={v0:.4f} "
                  + " ".join(f"{n}={params[n]:.3f}" for n in GLOBAL), flush=True)
    if warmup_dates:
        out = out[out["date"] >= dates[warmup_dates]].copy()
    return out
