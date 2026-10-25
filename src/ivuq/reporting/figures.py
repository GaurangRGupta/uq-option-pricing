"""Static figures for the Heston theo study (matplotlib, saved as PNG). Each function takes already-loaded frames and
writes one file into `outdir`; none reads results on its own, so the driver script controls what is shown.

Colours are fixed per theo so a reader can follow one model across figures (Okabe-Ito palette).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ivuq.pricing import black_scholes as bs
from ivuq.reporting.analysis import SPLIT, THEO_NAMES

COL = {"pinn": "#0072B2", "cpl1000": "#009E73", "fd": "#E69F00", "flat": "#CC79A7", "smile": "#7F7F7F",
       "h_slow": "#56B4E9", "random": "#D55E00"}
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "legend.frameon": False, "text.parse_math": False})
ORDER = ["pinn", "cpl1000", "fd", "flat", "smile"]


def _save(fig, outdir, name):
    p = Path(outdir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def iv_of(price, S, K, T, r, q):
    try:
        return bs.implied_vol(float(price), S, K, T, r, q, "put")
    except (ValueError, RuntimeError):
        return np.nan


# ---------------------------------------------------------------- the market: implied volatility surfaces
def iv_time_surface(puts: pd.DataFrame, sym: str, outdir, target_dte: int = 30):
    """Implied vol against moneyness S/K and calendar date at a constant maturity: on each date the two expiries
    around `target_dte` are interpolated linearly in total variance (iv^2 * days), so the surface does not jump when
    the nearest expiry rolls. Cells outside the quoted strike range of either expiry are left empty; a 3-day
    rolling median removes quote noise."""
    g = puts[puts["symbol"] == sym]
    grid = np.linspace(0.92, 1.10, 19)
    dates, rows = [], []
    for d, x in g.groupby("date"):
        e = x.groupby("expiry")["dte"].first().sort_values()
        lo, hi = e[e <= target_dte], e[e > target_dte]
        if not len(lo) or not len(hi):
            continue
        (e1, d1), (e2, d2) = (lo.index[-1], lo.iloc[-1]), (hi.index[0], hi.iloc[0])
        iv = []
        for ex in (e1, e2):
            y = x[x["expiry"] == ex].sort_values("m")
            iv.append(np.interp(grid, y["m"], y["vendor_iv"], left=np.nan, right=np.nan) if len(y) >= 5 else np.full(len(grid), np.nan))
        w = (target_dte - d1) / (d2 - d1)
        var = (1 - w) * iv[0] ** 2 * d1 + w * iv[1] ** 2 * d2
        dates.append(d)
        rows.append(np.sqrt(var / target_dte))
    Z = pd.DataFrame(rows).rolling(5, center=True, min_periods=1).median().T.rolling(3, center=True, min_periods=1).mean().T.to_numpy()
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    X, Y = np.meshgrid(grid, np.arange(len(dates)))
    ax.plot_surface(X, Y, np.ma.masked_invalid(Z), cmap="viridis", linewidth=0, antialiased=True, alpha=0.95)
    ticks = np.linspace(0, len(dates) - 1, 6).astype(int)
    ax.set_yticks(ticks)
    ax.set_yticklabels([pd.Timestamp(dates[i]).strftime("%b %y") for i in ticks])
    ax.set_xlabel("moneyness S/K")
    ax.set_zlabel("implied vol")
    ax.view_init(24, -58)
    ax.set_title(f"{sym}: implied volatility across strikes and time ({target_dte}-day constant maturity)")
    _save(fig, outdir, f"iv_surface_time_{sym}.png")
    return dates, grid, Z


def iv_term_surface(day: pd.DataFrame, sym: str, outdir, tag: str):
    """All quotes of one date: implied vol over moneyness and days to expiry."""
    day = day[(day["m"] >= 0.9) & (day["vendor_iv"] < 0.8)]   # deep in-the-money American puts invert to meaningless European vols
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(day["m"], day["dte"], day["vendor_iv"], c=day["vendor_iv"], cmap="viridis", s=8)
    ax.set_xlabel("moneyness S/K")
    ax.set_ylabel("days to expiry")
    ax.set_zlabel("implied vol")
    ax.view_init(22, -62)
    fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1, label="implied vol")
    ax.set_title(f"{sym}: implied volatility surface, {pd.Timestamp(day['date'].iloc[0]).date()} ({tag})")
    _save(fig, outdir, f"iv_surface_{tag}_{sym}.png")


def smile_fit(day: pd.DataFrame, sym: str, outdir, tag: str, theo_col: str = "theo", max_exp: int = 4):
    """For each expiry: implied vol of bid, ask, theo and the conformal band edges, across strikes.
    `day` needs bid, ask, theo, half_width, spot, strike, tau, r, q, expiry."""
    exps = sorted(day["expiry"].unique())
    pick = exps[:max_exp] if len(exps) <= max_exp else [exps[i] for i in np.linspace(0, len(exps) - 1, max_exp).astype(int)]
    fig, axs = plt.subplots(1, len(pick), figsize=(3.6 * len(pick), 3.6), sharey=False)
    axs = np.atleast_1d(axs)
    for ax, ex in zip(axs, pick):
        g = day[(day["expiry"] == ex) & (day["m"] >= 0.97)].sort_values("strike")   # European inversion is only meaningful for near-the-money and out-of-the-money puts (S/K >= 0.97); in-the-money American puts carry an early-exercise premium
        if len(g) < 3:
            continue
        f = lambda col: np.array([iv_of(v, r.spot, r.strike, r.tau, r.r, r.q) for v, r in zip(g[col], g.itertuples())])
        ax.fill_between(g["m"], f("bid"), f("ask"), color="#bbbbbb", alpha=0.6, label="market bid to ask")
        ax.plot(g["m"], f(theo_col), color=COL["pinn"], lw=1.8, label="model theo")
        lo = (g[theo_col] - g["half_width"]).clip(lower=0.01)
        hi = g[theo_col] + g["half_width"]
        ax.plot(g["m"], f_series(lo, g), color=COL["pinn"], lw=0.8, ls="--", label="conformal band")
        ax.plot(g["m"], f_series(hi, g), color=COL["pinn"], lw=0.8, ls="--")
        ax.set_title(f"{pd.Timestamp(ex).date()}  ({int(g['dte'].iloc[0])}d)")
        ax.set_xlabel("moneyness S/K")
        lo_, hi_ = np.nanmin(f("bid")), np.nanmax(f("ask"))
        if np.isfinite(lo_) and np.isfinite(hi_):
            ax.set_ylim(lo_ * 0.9, min(hi_ * 1.05, 1.5))
    axs[0].set_ylabel("implied vol (Black-Scholes inversion of the price)")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"{sym}: market quotes against the model theo, {pd.Timestamp(day['date'].iloc[0]).date()} ({tag})", y=1.02)
    _save(fig, outdir, f"smile_fit_{tag}_{sym}.png")


def f_series(price, g):
    return np.array([iv_of(v, r.spot, r.strike, r.tau, r.r, r.q) for v, r in zip(price, g.itertuples())])


def price_error_map(panel: pd.DataFrame, theo_col: str, outdir, name: str, title: str):
    """Mean absolute (mid - theo) in units of the error scale c, over moneyness x days to expiry."""
    d = panel.copy()
    d["z"] = (d["mid"] - d[theo_col]).abs() / d["scale"]
    mb = pd.cut(d["m"], [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25])
    tb = pd.cut(d["dte"], [0, 14, 21, 30, 45, 70])
    t = d.groupby([tb, mb], observed=False)["z"].median().unstack()
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    im = ax.imshow(t.to_numpy(), cmap="magma_r", aspect="auto", origin="lower")
    ax.set_xticks(range(t.shape[1]))
    ax.set_xticklabels([str(c) for c in t.columns], rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(t.shape[0]))
    ax.set_yticklabels([str(i) for i in t.index], fontsize=7)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7, color="white" if v > np.nanmax(t.to_numpy()) * 0.55 else "black")
    ax.grid(False)
    ax.set_xlabel("moneyness S/K")
    ax.set_ylabel("days to expiry")
    fig.colorbar(im, ax=ax, label="median |mid - theo| / c")
    ax.set_title(title)
    _save(fig, outdir, name)


def params_time(theo: dict[str, pd.DataFrame], boxes: dict, outdir, tag: str):
    """Calibrated Heston parameters per day; the shaded band is the box; points on its edge are 'pinned'."""
    names = [("kappa", "mean reversion kappa"), ("theta", "long-run variance theta"), ("xi", "vol of vol xi"),
             ("rho", "correlation rho"), ("v0", "spot variance v0"), ("fit_rmse", "fit RMS (half-spreads)")]
    fig, axs = plt.subplots(2, 3, figsize=(11, 5.6), sharex=True)
    for ax, (k, lab) in zip(axs.ravel(), names):
        for sym, t in theo.items():
            d = t.groupby("date")[k].first()
            ax.plot(d.index, d.values, lw=0.9, label=sym)
            if k in boxes[sym]:
                lo, hi = boxes[sym][k]
                pinned = ((d - lo).abs() < 0.01 * (hi - lo)) | ((d - hi).abs() < 0.01 * (hi - lo))
                ax.scatter(d.index[pinned], d.values[pinned], s=6, c="k", zorder=3)
        ax.set_title(lab)
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(f"Daily calibrated parameters ({tag}); black dots sit on the edge of the parameter box", y=1.0)
    fig.autofmt_xdate()
    _save(fig, outdir, f"params_time_{tag}.png")


def network_accuracy(evals: dict[str, dict], outdir):
    """MAE and p95 of the network against fresh exact-solver solves, $ per $100 strike."""
    syms = ["SPY", "AAPL", "NVDA"]
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.3))
    for ax, key, lab in zip(axs, ("mae", "p95"), ("mean absolute error", "95th percentile error")):
        w = 0.36
        for j, (tag, name, c) in enumerate((("cv_anch1000", "single network", COL["pinn"]), ("cpl1000", "coupled network", COL["cpl1000"]))):
            v = [evals[f"{s}_{tag}"][key] * 100 for s in syms]
            ax.bar(np.arange(3) + (j - 0.5) * w, v, w, color=c, label=name)
            for i, x in enumerate(v):
                ax.text(i + (j - 0.5) * w, x, f"{x:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(syms)
        ax.set_ylabel("cents per $100 strike")
        ax.set_title(lab)
    axs[0].legend()
    fig.suptitle("Network against fresh exact-solver prices (held-out parameters and states)", y=1.02)
    _save(fig, outdir, "network_accuracy.png")


# ---------------------------------------------------------------- trading: ablation and equity
def ablation_bars(ab: pd.DataFrame, outdir):
    """Mean daily net P&L with 95% block-bootstrap CI: theo x rule, development vs test, same close vs next close."""
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.2), sharey="row")
    labels = [THEO_NAMES[t] for t in ORDER]
    for r, lag in enumerate((0, 1)):
        for c, per in enumerate(("development", "test")):
            ax = axs[r, c]
            for j, (rule, hatch, alpha) in enumerate((("spread", "//", 0.45), ("conformal", None, 1.0))):
                v = []
                for t in ORDER:
                    x = ab[(ab.model == t) & (ab.rule == rule) & (ab.lag == lag) & (ab.period == per)]
                    v.append(x.iloc[0] if len(x) else None)
                m = [x["total"] / x["days"] if x is not None and x["days"] else np.nan for x in v]
                lo = [(x["ci_lo"]) if x is not None else np.nan for x in v]
                hi = [(x["ci_hi"]) if x is not None else np.nan for x in v]
                xs = np.arange(len(ORDER)) + (j - 0.5) * 0.38
                ax.bar(xs, m, 0.38, color=[COL[t] for t in ORDER], alpha=alpha, hatch=hatch, edgecolor="white",
                       label="no band (any edge after spread)" if rule == "spread" else "with conformal band")
                ax.errorbar(xs, m, yerr=[np.array(m) - np.array(lo), np.array(hi) - np.array(m)], fmt="none", ecolor="k", lw=0.8, capsize=2)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(range(len(ORDER)))
            ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=7)
            ax.set_title(f"{per} period, fills at the {'same close' if lag == 0 else 'next close'}")
            if c == 0:
                ax.set_ylabel("mean daily net P&L ($)")
    h, l = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend([plt.Rectangle((0, 0), 1, 1, fc="#888", alpha=0.45, hatch="//"), plt.Rectangle((0, 0), 1, 1, fc="#888")],
                     ["no band (any edge after spread)", "with conformal band"], fontsize=7, loc="upper left")
    fig.suptitle("What the conformal band does: each theo traded with and without it (bars: mean daily P&L, whiskers: 95% bootstrap CI)", y=1.0)
    fig.tight_layout()
    _save(fig, outdir, "ablation_band.png")


def equity_curves(trades: pd.DataFrame, outdir, lag: int = 0):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, rule in zip(axs, ("conformal", "spread")):
        for t in ORDER:
            g = trades[(trades.model == t) & (trades.rule == rule) & (trades.lag == lag)]
            if not len(g):
                continue
            c = g.groupby("exit_date")["net_pnl"].sum().sort_index().cumsum()
            ax.plot(c.index, c.values, color=COL[t], lw=1.4, label=THEO_NAMES[t])
        ax.axvline(SPLIT, color="k", ls=":", lw=0.9)
        ax.text(SPLIT, ax.get_ylim()[1], " test period", va="top", fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title("with conformal band" if rule == "conformal" else "no band (any edge after spread)")
    axs[0].set_ylabel("cumulative net P&L ($)")
    axs[0].legend(fontsize=7)
    fig.suptitle(f"Cumulative P&L, delta-hedged, fills at the {'same' if lag == 0 else 'next'} close, horizon 3 days", y=1.02)
    fig.autofmt_xdate()
    _save(fig, outdir, f"equity_lag{lag}.png")


def lag_decay(ab: pd.DataFrame, outdir):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = ab[(ab.rule == "conformal") & (ab.period == "test")]
    for j, t in enumerate(ORDER):
        for k, lag in enumerate((0, 1)):
            r = x[(x.model == t) & (x.lag == lag)]
            if len(r):
                ax.bar(j + (k - 0.5) * 0.38, r.iloc[0]["total"], 0.38, color=COL[t], alpha=1.0 if lag == 0 else 0.4,
                       hatch=None if lag == 0 else "//", edgecolor="white")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([THEO_NAMES[t] for t in ORDER], rotation=18, ha="right", fontsize=7)
    ax.set_ylabel("total net P&L, test period ($)")
    ax.set_title("Execution delay: solid = fill at the signal close, hatched = fill at the next close")
    _save(fig, outdir, "lag_decay.png")


def breakdown(trades: pd.DataFrame, outdir, theo: str = "pinn", lag: int = 0):
    g = trades[(trades.model == theo) & (trades.rule == "conformal") & (trades.lag == lag)].copy()
    g["dte_at_entry"] = (g["expiry"] - g["entry_date"]).dt.days
    g["tenor"] = pd.cut(g["dte_at_entry"], [0, 14, 30, 45, 80], labels=["<=14d", "15-30d", "31-45d", ">45d"])
    g["side_l"] = np.where(g["side"] > 0, "buy (long)", "sell (short)")
    g["moneyness"] = g["bucket"].map({0: "ITM", 1: "ITM", 2: "ATM", 3: "ATM", 4: "OTM", 5: "OTM"})   # bucket = 2 * (S/K < 0.95, 0.95 to 1.05, > 1.05) + (DTE >= 21)
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.3), sharey=True)
    for ax, key in zip(axs, ("symbol", "side_l", "tenor", "moneyness")):
        for per, hatch in (("development", "//"), ("test", None)):
            s = g[g.period == per].groupby(key, observed=True)["net_pnl"].agg(["mean", "size"])
            idx = {k: i for i, k in enumerate(sorted(g[key].dropna().unique(), key=str))}
            xs = np.array([idx[k] for k in s.index]) + (0.2 if per == "test" else -0.2)
            ax.bar(xs, s["mean"], 0.38, color=COL[theo], alpha=0.45 if hatch else 1.0, hatch=hatch, edgecolor="white")
            for x, (m, n) in zip(xs, s.itertuples(index=False)):
                ax.text(x, m, f"n={n}", ha="center", va="bottom" if m >= 0 else "top", fontsize=6)
        ax.set_xticks(range(len(idx)))
        ax.set_xticklabels(list(idx), fontsize=7)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(key.replace("_l", "").replace("dte_at_entry", "tenor"))
    axs[0].set_ylabel("mean net P&L per trade ($)")
    fig.suptitle(f"Where the P&L comes from: {THEO_NAMES[theo]} with band, same close (hatched = development, solid = test)", y=1.04)
    _save(fig, outdir, f"breakdown_{theo}_lag{lag}.png")


def cost_heatmap(cg: pd.DataFrame, outdir, theo="pinn", lag=0):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, per in zip(axs, ("development", "test")):
        t = cg[(cg.model == theo) & (cg.rule == "conformal") & (cg.lag == lag) & (cg.period == per)]
        pv = t.pivot(index="commission", columns="hedge_mult", values="mean")
        im = ax.imshow(pv.to_numpy(), cmap="RdBu", vmin=-np.nanmax(np.abs(pv.to_numpy())), vmax=np.nanmax(np.abs(pv.to_numpy())), origin="lower", aspect="auto")
        ax.set_xticks(range(pv.shape[1]))
        ax.set_xticklabels([f"{c:g}x" for c in pv.columns])
        ax.set_yticks(range(pv.shape[0]))
        ax.set_yticklabels([f"${c:g}" for c in pv.index])
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.iat[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(pv.iat[i, j]) > 0.5 * np.nanmax(np.abs(pv.to_numpy())) else "black")
        ax.grid(False)
        ax.set_xlabel("stock-hedging cost (multiple of 1 bp)")
        ax.set_ylabel("commission per contract per side")
        ax.set_title(per)
    fig.suptitle(f"Cost sensitivity: mean net P&L per trade ($), {THEO_NAMES[theo]} with band", y=1.03)
    _save(fig, outdir, f"cost_sensitivity_{theo}.png")


def conformal_coverage(diag: pd.DataFrame, outdir, alpha=0.10):
    d = diag[diag.variant == "h_full"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.5))
    for band, c, ls in (("conformal", COL["pinn"], "-"), ("conformal_aci", COL["fd"], "-")):
        x = d[d.band == band].copy()
        x["miss"] = x["miss_rate"] * x["n"]
        lab = "fixed alpha" if band == "conformal" else "adaptive alpha"
        s = x.groupby("symbol")[["miss", "n"]].sum()
        axs[0].bar(np.arange(len(s)) + (0.2 if band != "conformal" else -0.2), s["miss"] / s["n"], 0.38, color=c, label=lab)
        b = x.groupby("bucket")[["miss", "n"]].sum()
        axs[1].plot(b.index, b["miss"] / b["n"], "o-", color=c, label=lab)
        m = x.groupby("month")[["miss", "n"]].sum()
        axs[2].plot(range(len(m)), m["miss"] / m["n"], "o-", color=c, label=lab)
        axs[2].set_xticks(range(len(m)))
        axs[2].set_xticklabels(m.index, rotation=60, fontsize=6)
        if band == "conformal":
            axs[0].set_xticks(range(len(s)))
            axs[0].set_xticklabels(s.index)
    for ax, t in zip(axs, ("by underlying", "by moneyness x tenor bucket", "by month")):
        ax.axhline(alpha, color="k", ls="--", lw=0.9)
        ax.set_title(t)
        ax.set_ylim(0, 0.30)
    axs[0].set_ylabel("realised miss rate (target 10%)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Does the past-only conformal band cover? Fraction of quotes outside the band, Heston PINN", y=1.03)
    _save(fig, outdir, "conformal_coverage.png")


def gap_closure(panel: pd.DataFrame, cols: dict[str, str], outdir):
    """Does the market move toward the theo? x = (theo - mid)/c today (positive: the market is cheap), y = the next-trading-day
    change of the same contract's mid, net of the stock move at the vendor delta, over c. The slope k is the fraction of the gap
    that the market itself closes by the next close; k = 0 means no reversion, k = 1 full reversion. Only consecutive trading
    days (at most 4 calendar days apart) with |x|, |y| < 15 are used. Returns {name: (slope, standard error, n)}."""
    fig, axs = plt.subplots(1, len(cols), figsize=(4.0 * len(cols), 3.6), sharey=True)
    p = panel.sort_values("date").copy()
    key = ["symbol", "expiry", "strike"]
    p["next_date"] = p.groupby(key)["date"].shift(-1)
    p["next_mid"] = p.groupby(key)["mid"].shift(-1)
    sp = p.groupby(["symbol", "date"])["spot"].first()
    p["next_spot"] = [sp.get((s_, d_), np.nan) for s_, d_ in zip(p["symbol"], p["next_date"])]
    p["y"] = (p["next_mid"] - p["mid"] - p["delta"] * (p["next_spot"] - p["spot"])) / p["scale"]
    res = {}
    for ax, (name, col) in zip(np.atleast_1d(axs), cols.items()):
        p["x"] = (p[col] - p["mid"]) / p["scale"]
        d = p.dropna(subset=["y", "x"])
        d = d[((d["next_date"] - d["date"]).dt.days <= 4) & (d["x"].abs() < 15) & (d["y"].abs() < 15)]
        X = np.column_stack([np.ones(len(d)), d["x"]])
        beta, *_ = np.linalg.lstsq(X, d["y"].to_numpy(), rcond=None)
        resid = d["y"].to_numpy() - X @ beta
        se = float(np.sqrt(resid.var(ddof=2) / (len(d) * d["x"].var(ddof=0))))
        bins = pd.qcut(d["x"], 30, duplicates="drop")
        bm = d.groupby(bins, observed=True).agg(x=("x", "mean"), y=("y", "mean"))
        ax.scatter(d["x"], d["y"], s=2, alpha=0.08, color=COL.get(name, "k"))
        ax.plot(bm["x"], bm["y"], "o-", color="k", ms=3, lw=1)
        ax.plot([-8, 8], [-8, 8], ":", color="gray", lw=0.8)
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        ax.set_title(f"{THEO_NAMES.get(name, name)}: k = {beta[1]:.2f} (se {se:.2f})")
        ax.set_xlabel("gap today, (theo - mid)/c")
        res[name] = (float(beta[1]), se, int(len(d)))
    np.atleast_1d(axs)[0].set_ylabel("next-day hedged mid change / c")
    fig.suptitle("Does the market move toward the theo by the next close? dotted line: full reversion (k = 1)", y=1.03)
    _save(fig, outdir, "gap_closure.png")
    return res


def failure_panel(trades: pd.DataFrame, outdir, theo="pinn", lag=0):
    g = trades[(trades.model == theo) & (trades.rule == "conformal") & (trades.lag == lag)].copy()
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.6))
    a = axs[0, 0]
    a.hist(g["net_pnl"].clip(-600, 600), bins=60, color=COL[theo])
    q = g["net_pnl"].quantile([0.05, 0.5, 0.95])
    for v in q:
        a.axvline(v, color="k", lw=0.7, ls=":")
    a.set_title(f"P&L per trade: median ${q.iloc[1]:.0f}, 5% ${q.iloc[0]:.0f}, 95% ${q.iloc[2]:.0f}")
    a.set_xlabel("net P&L per trade ($, clipped at +/-600)")
    b = axs[0, 1]
    m = g.copy()
    m["month"] = m["exit_date"].dt.to_period("M").astype(str)
    s = m.groupby("month")["net_pnl"].sum()
    b.bar(range(len(s)), s.values, color=np.where(s.values >= 0, COL[theo], COL["random"]))
    b.set_xticks(range(len(s)))
    b.set_xticklabels(s.index, rotation=60, fontsize=7)
    b.axvline(list(s.index).index(str(SPLIT.to_period("M"))) - 0.5, color="k", ls=":", lw=0.9)
    b.set_title("Monthly net P&L: the edge is not stable across regimes")
    c = axs[1, 0]
    bins = pd.qcut(g["excess"], 8, duplicates="drop")
    bm = g.groupby(bins, observed=True).agg(x=("excess", "mean"), y=("net_pnl", "mean"), se=("net_pnl", lambda v: v.std() / np.sqrt(len(v))))
    c.errorbar(bm["x"], bm["y"], yerr=1.96 * bm["se"], fmt="o-", color=COL[theo], capsize=2)
    c.axhline(0, color="k", lw=0.6)
    c.set_xlabel("signal strength: (edge - band) / c")
    c.set_ylabel("mean net P&L ($)")
    c.set_title("Do larger signals earn more?")
    d = axs[1, 1]
    comp = pd.Series({"entry edge (x100)": 100 * g["edge"].mean(), "hedged gross": (g["opt_pnl"] + g["hedge_pnl"]).mean(),
                      "hedge costs": -g["hedge_cost"].mean(), "commissions": -g["commission"].mean(), "net": g["net_pnl"].mean()})
    d.bar(range(len(comp)), comp.values, color=[COL["cpl1000"] if v >= 0 else COL["random"] for v in comp.values])
    d.set_xticks(range(len(comp)))
    d.set_xticklabels(comp.index, fontsize=7)
    d.axhline(0, color="k", lw=0.6)
    for i, v in enumerate(comp.values):
        d.text(i, v, f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    d.set_title("Average trade: edge paid at entry, what the hedged position earned, costs")
    fig.suptitle(f"Failure anatomy: {THEO_NAMES[theo]} with band, fills at the {'same' if lag == 0 else 'next'} close", y=1.0)
    fig.tight_layout()
    _save(fig, outdir, f"failure_{theo}_lag{lag}.png")


def random_vs_model(trades: pd.DataFrame, ctrl: pd.DataFrame, outdir):
    """Total P&L of the model on the full sample against 30 random-signal runs with the same trade count."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for j, (t, var) in enumerate((("pinn", "h_full"), ("cpl1000", "h_full"), ("fd", "h_full"))):
        for k, lag in enumerate((0, 1)):
            g = trades[(trades.model == t) & (trades.rule == "conformal") & (trades.lag == lag)]
            c = ctrl[(ctrl.tag == t) & (ctrl.variant == var) & (ctrl.lag == lag)]
            x = j * 2.4 + k
            ax.bar(x, g["net_pnl"].sum(), 0.8, color=COL[t], alpha=1 if lag == 0 else 0.4)
            if len(c):
                ax.errorbar(x, c.iloc[0]["total_net_mean"], yerr=[[c.iloc[0]["total_net_mean"] - c.iloc[0]["total_net_p05"]],
                            [c.iloc[0]["total_net_p95"] - c.iloc[0]["total_net_mean"]]], fmt="D", color="k", capsize=3, ms=4)
    ax.set_xticks([0.5, 2.9, 5.3])
    ax.set_xticklabels([THEO_NAMES[t] for t in ("pinn", "cpl1000", "fd")], fontsize=8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("total net P&L, whole sample ($)")
    ax.set_title("Model (bars; solid = same close, light = next close) against 30 random-signal runs (diamond, 5-95%)")
    _save(fig, outdir, "model_vs_random.png")


def signal_map(sig: pd.DataFrame, outdir, tag: str, title: str):
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, sym in zip(axs, ("SPY", "AAPL", "NVDA")):
        g = sig[sig.symbol == sym]
        for act, c, mk in (("HOLD", "#cccccc", "o"), ("BUY", COL["cpl1000"], "^"), ("SELL", COL["random"], "v")):
            x = g[g.action == act]
            ax.scatter(x["m"], x["dte"], s=18 if act != "HOLD" else 8, c=c, marker=mk, label=f"{act} ({len(x)})")
        ax.set_title(sym)
        ax.set_xlabel("moneyness S/K")
        ax.legend(fontsize=7)
    axs[0].set_ylabel("days to expiry")
    fig.suptitle(title, y=1.03)
    _save(fig, outdir, f"signal_map_{tag}.png")


def live_shift(hist: pd.DataFrame, live: pd.DataFrame, outdir):
    """Historical (DoltHub) panel against the live Yahoo snapshot: quote density, spreads, coverage."""
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    syms = ["SPY", "AAPL", "NVDA"]
    dens_h = hist.groupby(["symbol", "date"]).size().groupby("symbol").mean()
    dens_l = live.groupby("symbol").size()
    axs[0].bar(np.arange(3) - 0.2, [dens_h[s] for s in syms], 0.38, color="#999999", label="DoltHub history")
    axs[0].bar(np.arange(3) + 0.2, [dens_l[s] for s in syms], 0.38, color=COL["pinn"], label="Yahoo live")
    axs[0].set_yscale("log")
    axs[0].set_title("put quotes per day")
    axs[1].bar(np.arange(3) - 0.2, [hist[hist.symbol == s]["spread"].median() for s in syms], 0.38, color="#999999")
    axs[1].bar(np.arange(3) + 0.2, [live[live.symbol == s]["spread"].median() for s in syms], 0.38, color=COL["pinn"])
    axs[1].set_title("median bid-ask spread ($)")
    cov_l = [live[(live.symbol == s) & np.isfinite(live["half_width"])]["covered"].mean() for s in syms]
    axs[2].bar(np.arange(3) + 0.2, cov_l, 0.38, color=COL["pinn"])
    axs[2].axhline(0.9, color="k", ls="--", lw=0.9)
    axs[2].set_title("band coverage on the live day (target 0.90)")
    axs[2].set_ylim(0, 1)
    for ax in axs:
        ax.set_xticks(range(3))
        ax.set_xticklabels(syms)
    axs[0].legend(fontsize=7)
    fig.suptitle("Transfer to the live chain: the band was calibrated on a sparser, wider-spread source", y=1.03)
    _save(fig, outdir, "live_shift.png")


def live_tags(live: dict[str, pd.DataFrame], outdir):
    """Same live day, each theo: band coverage, mean absolute error against the mid, and number of signals per underlying."""
    syms = ["SPY", "AAPL", "NVDA"]
    tags = [t for t in ("pinn", "cpl1000", "fd") if t in live]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    w = 0.8 / len(tags)
    for j, t in enumerate(tags):
        d = live[t]
        x = np.arange(3) + (j - (len(tags) - 1) / 2) * w
        cov = [d[(d.symbol == s) & np.isfinite(d["half_width"])]["covered"].mean() for s in syms]
        mae = [(d[d.symbol == s]["theo"] - d[d.symbol == s]["mid"]).abs().mean() for s in syms]
        nsig = [d[(d.symbol == s) & d["action"].isin(["BUY", "SELL"])].shape[0] / max(d[d.symbol == s].shape[0], 1) for s in syms]
        axs[0].bar(x, cov, w, color=COL[t], label=THEO_NAMES[t])
        axs[1].bar(x, mae, w, color=COL[t])
        axs[2].bar(x, nsig, w, color=COL[t])
    axs[0].axhline(0.9, color="k", ls="--", lw=0.9)
    axs[0].set_title("band coverage (target 0.90)")
    axs[1].set_title("mean |theo - mid| ($)")
    axs[2].set_title("share of quotes flagged BUY or SELL")
    for ax in axs:
        ax.set_xticks(range(3))
        ax.set_xticklabels(syms)
    axs[0].legend(fontsize=7)
    fig.suptitle("Live chain, one day: the exact solver (wide parameter box) fits the dense chain better than the networks (narrow box)", y=1.03)
    _save(fig, outdir, "live_tags.png")


def trade_example(trades: pd.DataFrame, panel: pd.DataFrame, outdir, theo="pinn", which="best"):
    """The life of one trade: bid, ask and mid of the contract from entry to exit, the theo and band at entry, the P&L pieces."""
    g = trades[(trades.model == theo) & (trades.rule == "conformal") & (trades.lag == 0)].copy()
    g = g.sort_values("net_pnl", ascending=(which != "best"))
    for r in g.itertuples():
        q = panel[(panel.symbol == r.symbol) & (panel.expiry == r.expiry) & (panel.strike == r.strike)].sort_values("date")
        q = q[(q.date >= r.entry_date - pd.Timedelta(days=4)) & (q.date <= r.exit_date + pd.Timedelta(days=4))]
        if len(q) >= 4:
            break
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = np.arange(len(q))
    ax.fill_between(x, q["bid"], q["ask"], color="#bbbbbb", alpha=0.7, label="market bid to ask")
    ax.plot(x, q["mid"], color="k", lw=1.2, label="mid")
    ie = int(np.where(q["date"].to_numpy() == np.datetime64(r.entry_date))[0][0])
    ix = int(np.where(q["date"].to_numpy() == np.datetime64(r.exit_date))[0][0])
    ax.axhline(r.theo, color=COL[theo], lw=1.6, label=f"theo at entry ${r.theo:.2f}")
    ax.axhspan(r.theo - (r.edge - r.excess * r.scale), r.theo + (r.edge - r.excess * r.scale), color=COL[theo], alpha=0.12, label="conformal band")
    ax.scatter([ie], [r.entry], marker="^" if r.side > 0 else "v", s=80, c=COL["cpl1000"], zorder=4, label=f"{'buy at ask' if r.side > 0 else 'sell at bid'} ${r.entry:.2f}")
    ax.scatter([ix], [r.exit], marker="s", s=60, c=COL["random"], zorder=4, label=f"close at ${r.exit:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([pd.Timestamp(d).strftime("%d %b") for d in q["date"]], fontsize=7)
    ax.set_ylabel("put price ($)")
    ax.set_title(f"{r.symbol} {pd.Timestamp(r.expiry).date()} put K={r.strike:g}: option ${r.opt_pnl:.0f}, hedge ${r.hedge_pnl:.0f}, costs -${r.hedge_cost + r.commission:.0f}, net ${r.net_pnl:.0f}", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    _save(fig, outdir, f"trade_example_{theo}_{which}.png")
