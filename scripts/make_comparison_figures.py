"""Figures for the PINN vs ML/DL comparison (American put first).

  python scripts/make_comparison_figures.py --results results/baselines --out results/figures
  python scripts/make_comparison_figures.py --skip-showcase        # CSV figures only (no training)

Part 1 reads the CSVs from run_baseline_comparison.py:
  fig1_price_error_vs_labels   price error against number of labelled prices, per regime
  fig2_boundary_error_vs_labels  exercise boundary error, same layout
  fig3_iv_error_vs_labels      implied vol error, same layout
  fig4_arbitrage_violations    fraction of grid points breaking no-arbitrage rules
  fig5_extrapolation           error inside vs outside the labelled region
  fig6_cost_vs_accuracy        training seconds against price error
  fig7_european_*              European accuracy, Greeks and PDE residual

Part 2 trains one PINN and the main baselines and draws what the tables cannot show:
  fig8_boundary_curves         each model's exercise boundary against the tree's
  fig9_error_maps              where in (moneyness, maturity) each model is wrong
  fig10_price_slice            price vs moneyness at fixed maturity, near the kink

Every figure plots whatever the data says; nothing here assumes which model wins.
"""

from __future__ import annotations

import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from ivuq.baselines import (
    EXTRAPOLATION_BOX,
    FULL_BOX,
    MarketParams,
    make_baselines,
    pinn_predictor,
    reference_u,
    sample_labelled,
)
from ivuq.baselines.metrics import BOUNDARY_M, BOUNDARY_TAU, exercise_boundary

PALETTE = {
    "pinn_n2": "#d62728", "pinn_n1": "#d62728", "pinn_n0": "#ff9896",
    "poly_ridge": "#9ecae1", "kernel_ridge": "#3182bd", "random_forest": "#6baed6", "grad_boosting": "#08519c",
    "mlp_tanh": "#fd8d3c", "mlp_relu": "#fdae6b", "resnet_silu": "#a1d99b",
    "fourier_mlp": "#756bb1", "picnn": "#31a354", "bs_iv_net": "#8c564b",
}
LABEL = {
    "pinn_n2": "PINN (American)", "pinn_n1": "PINN (N1)", "pinn_n0": "no-physics control (N0)",
    "poly_ridge": "polynomial ridge", "kernel_ridge": "kernel ridge", "random_forest": "random forest",
    "grad_boosting": "gradient boosting", "mlp_tanh": "MLP (tanh, PINN-shaped)", "mlp_relu": "MLP (ReLU)",
    "resnet_silu": "residual net", "fourier_mlp": "Fourier-feature net", "picnn": "input-convex net",
    "bs_iv_net": "BS-embedded IV net",
}
REGIMES = [("full", "Labels everywhere"), ("extrapolate", "Labels near the money only"), ("noisy", "Noisy labels (\$0.50)")]
DPI = 170


def _color(m: str) -> str:
    return PALETTE.get(m, "#666666")


def _load(results: Path, style: str, option_type: str) -> pd.DataFrame | None:
    frames = [pd.read_csv(p) for p in sorted(results.glob(f"{style}_{option_type}_*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else None


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.png")


def _vs_labels(df: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path, name: str, logy: bool = True) -> None:
    regimes = [(k, t) for k, t in REGIMES if k in set(df["regime"])]
    if not regimes or df[metric].isna().all():
        return
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.2 * len(regimes), 4.4), sharey=True, squeeze=False)
    for ax, (reg, reg_title) in zip(axes[0], regimes):
        d = df[df.regime == reg]
        for model, g in d[d.family != "pinn"].groupby("model"):
            c = g.groupby("n_train")[metric].mean()
            ax.plot(c.index, c.values, marker="o", ms=3.5, lw=1.4, color=_color(model), label=LABEL.get(model, model))
        for model, g in d[d.family == "pinn"].groupby("model"):
            v = g[metric].mean()
            ax.axhline(v, color=_color(model), lw=2.6, ls="--" if model == "pinn_n0" else "-", label=LABEL.get(model, model) + " (0 labels)")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_title(reg_title, fontsize=11)
        ax.set_xlabel("number of labelled prices given to the baselines")
        ax.grid(alpha=0.25, which="both")
    axes[0][0].set_ylabel(ylabel)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle(title, fontsize=13, y=1.02)
    _save(fig, out, name)


def _bars(ax, d: pd.DataFrame, cols: list[str], titles: list[str], models: list[str]) -> None:
    w = 0.8 / len(cols)
    x = np.arange(len(models))
    for i, (c, t) in enumerate(zip(cols, titles)):
        vals = [d[d.model == m][c].mean() for m in models]
        ax.bar(x + i * w, vals, w, label=t)
    ax.set_xticks(x + w * (len(cols) - 1) / 2)
    ax.set_xticklabels([LABEL.get(m, m) for m in models], rotation=40, ha="right", fontsize=8)


def fig_arbitrage(df: pd.DataFrame, out: Path) -> None:
    d = df[(df.regime == "full") & ((df.n_train == df.n_train.max()) | (df.family == "pinn"))]
    if d.empty:
        return
    models = list(d.model.unique())
    fig, ax = plt.subplots(figsize=(10, 4.4))
    _bars(ax, d, ["viol_bound", "viol_calendar", "viol_convex"],
          ["price below intrinsic value", "price falls with more time", "price not convex in spot"], models)
    ax.set_ylabel("fraction of test points violating the rule")
    ax.set_title(f"American put: no-arbitrage violations (baselines given {int(d.n_train.max())} labels)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out, "fig4_arbitrage_violations")


def fig_extrapolation(df: pd.DataFrame, out: Path) -> None:
    d = df[df.regime == "extrapolate"]
    if d.empty:
        return
    n = 1000 if 1000 in set(d.n_train) else d.n_train.max()
    d = d[(d.n_train == n) | (d.family == "pinn")]
    models = list(d.model.unique())
    fig, ax = plt.subplots(figsize=(10, 4.4))
    _bars(ax, d, ["mae_inside_usd", "mae_outside_usd"], ["inside the labelled region", "outside the labelled region"], models)
    ax.set_yscale("log")
    ax.set_ylabel("price error (\$ on a \$100 strike)")
    ax.set_title(f"American put: what happens away from the data (baselines given {int(n)} labels near the money)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25, which="both")
    _save(fig, out, "fig5_extrapolation")


def fig_cost(df: pd.DataFrame, out: Path) -> None:
    d = df[(df.regime == "full") & ((df.n_train == df.n_train.max()) | (df.family == "pinn"))]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for model, g in d.groupby("model"):
        ax.scatter(g.fit_seconds.mean(), g.mae_usd.mean(), s=70, color=_color(model), zorder=3)
        ax.annotate(LABEL.get(model, model), (g.fit_seconds.mean(), g.mae_usd.mean()), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training time (seconds)")
    ax.set_ylabel("price error (\$ on a \$100 strike)")
    ax.set_title("American put: cost against accuracy (labels everywhere)")
    ax.grid(alpha=0.25, which="both")
    _save(fig, out, "fig6_cost_vs_accuracy")


def fig_european(df: pd.DataFrame, out: Path, option_type: str) -> None:
    d = df[(df.regime == "full") & ((df.n_train == df.n_train.max()) | (df.family == "pinn"))]
    if d.empty:
        return
    models = list(d.model.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (c, t) in zip(axes, [("mae_usd", "price error (\$)"), ("gamma_mae", "gamma error"), ("pde_rms", "Black-Scholes PDE residual (rms)")]):
        vals = [d[d.model == m][c].mean() for m in models]
        ax.bar(range(len(models)), vals, color=[_color(m) for m in models])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([LABEL.get(m, m) for m in models], rotation=45, ha="right", fontsize=7)
        ax.set_yscale("log")
        ax.set_title(t, fontsize=10)
        ax.grid(axis="y", alpha=0.25, which="both")
    fig.suptitle(f"European {option_type}: accuracy and derivative quality (baselines given {int(d.n_train.max())} labels)", y=1.03)
    _save(fig, out, f"fig7_european_{option_type}")


def part1(results: Path, out: Path) -> None:
    a = _load(results, "american", "put")
    if a is not None:
        _vs_labels(a, "mae_usd", "price error (\$, \$100 strike)", "American put: price error against number of labels", out, "fig1_price_error_vs_labels")
        _vs_labels(a, "boundary_mae_usd", "exercise boundary error (\$, \$100 strike)", "American put: exercise boundary error against number of labels", out, "fig2_boundary_error_vs_labels")
        _vs_labels(a, "iv_mae_pts", "implied vol error (vol points)", "American put: implied volatility error against number of labels", out, "fig3_iv_error_vs_labels")
        fig_arbitrage(a, out)
        fig_extrapolation(a, out)
        fig_cost(a, out)
    for opt in ("put", "call"):
        e = _load(results, "european", opt)
        if e is not None:
            _vs_labels(e, "mae_usd", "price error (\$, \$100 strike)", f"European {opt}: price error against number of labels", out, f"fig7_european_{opt}_vs_labels")
            fig_european(e, out, opt)


def _grid(m_lo: float, m_hi: float, tau_lo: float, tau_hi: float, nm: int, nt: int):
    mm, tt = np.meshgrid(np.linspace(m_lo, m_hi, nm), np.linspace(tau_lo, tau_hi, nt))
    return mm, tt


def part2(out: Path, fast: bool) -> None:
    from ivuq.pinn import AmericanPINN, AmericanPINNConfig

    p = MarketParams()
    kw = dict(hidden_layers=3, hidden_width=24, epochs=1500, n_interior=1000, n_terminal=200, n_boundary=100, lr_decay_every=700) if fast else {}
    print("training the American PINN for the showcase figures", flush=True)
    pinn = AmericanPINN(AmericanPINNConfig(option_type="put", r=p.r, q=p.q, sigma=p.sigma, **kw)).fit()
    predictors = {"pinn_n2": pinn_predictor(pinn)}
    names = ["kernel_ridge", "random_forest", "mlp_tanh", "resnet_silu", "picnn", "bs_iv_net"]

    setups = {}
    for tag, box, n in (("full", FULL_BOX, 1000), ("extrapolate", EXTRAPOLATION_BOX, 300)):
        X, u = sample_labelled("american", "put", n, box, p, seed=0)
        preds = dict(predictors)
        for name, model in make_baselines(seed=0, fast=fast, option_type="put", params=p).items():
            if name in names:
                model.fit(X, u)
                preds[name] = (lambda mo: (lambda m, t: mo.predict(np.column_stack([m, t]))))(model)
        setups[tag] = (box, n, preds)
        print(f"  fitted baselines for {tag}", flush=True)

    # fig8: exercise boundary curves
    tau_dense = BOUNDARY_TAU
    mm, tt = np.meshgrid(BOUNDARY_M, BOUNDARY_TAU)
    u_ref_b = reference_u("american", "put", mm.ravel(), tt.ravel(), p).reshape(mm.shape)
    ref_b = exercise_boundary(lambda m, t: u_ref_b.ravel()[: m.size])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, tag in zip(axes, ("full", "extrapolate")):
        box, n, preds = setups[tag]
        ax.plot(tau_dense, ref_b, color="black", lw=3, label="binomial tree (reference)")
        for name, f in preds.items():
            b = exercise_boundary(f)
            ax.plot(tau_dense, b, marker="o", ms=3, lw=2.4 if name == "pinn_n2" else 1.2, color=_color(name), label=LABEL[name])
        ax.axvspan(box.tau_lo, box.tau_hi, color="grey", alpha=0.08)
        ax.set_title("labels everywhere" if tag == "full" else "labels only for maturity up to 0.5 (shaded)", fontsize=11)
        ax.set_xlabel("time to expiry (years)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("exercise boundary (moneyness S/K)")
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("American put: the early-exercise boundary each model implies", y=1.02)
    _save(fig, out, "fig8_boundary_curves")

    # fig9: error maps in the extrapolation regime
    box, n, preds = setups["extrapolate"]
    gm, gt = _grid(0.5, 1.6, 0.05, 1.0, 45, 30)
    u_ref = reference_u("american", "put", gm.ravel(), gt.ravel(), p).reshape(gm.shape)
    show = ["pinn_n2", "kernel_ridge", "mlp_tanh", "picnn", "bs_iv_net", "random_forest"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharex=True, sharey=True)
    vmax = 5.0
    for ax, name in zip(axes.ravel(), show):
        err = 100 * np.abs(preds[name](gm.ravel(), gt.ravel()).reshape(gm.shape) - u_ref)
        im = ax.pcolormesh(gm, gt, np.minimum(err, vmax), cmap="magma_r", vmin=0, vmax=vmax, shading="auto")
        ax.add_patch(Rectangle((box.m_lo, box.tau_lo), box.m_hi - box.m_lo, box.tau_hi - box.tau_lo, fill=False, ec="cyan", lw=1.6))
        ax.set_title(f"{LABEL[name]}  (mean \${err.mean():.2f})", fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel("moneyness S/K")
    for ax in axes[:, 0]:
        ax.set_ylabel("time to expiry")
    fig.colorbar(im, ax=axes, shrink=0.8, label="price error (\$ on a \$100 strike, capped at \$5)")
    fig.suptitle("American put: where each model is wrong. Cyan box = the only region with labelled prices", y=0.99)
    _save(fig, out, "fig9_error_maps")

    # fig10: price slice
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    ms = np.linspace(0.5, 1.5, 200)
    for ax, tag in zip(axes, ("full", "extrapolate")):
        box, n, preds = setups[tag]
        tau = np.full_like(ms, 0.4)
        ref = reference_u("american", "put", ms, tau, p)
        ax.plot(ms, 100 * ref, color="black", lw=3, label="binomial tree (reference)")
        ax.plot(ms, 100 * np.maximum(1 - ms, 0), color="grey", ls=":", lw=1.2, label="intrinsic value")
        for name, f in preds.items():
            ax.plot(ms, 100 * f(ms, tau), lw=2.2 if name == "pinn_n2" else 1.1, color=_color(name), label=LABEL[name])
        if tag == "extrapolate":
            ax.axvspan(box.m_lo, box.m_hi, color="grey", alpha=0.10)
        ax.set_title("labels everywhere" if tag == "full" else "labels only near the money (shaded)", fontsize=11)
        ax.set_xlabel("moneyness S/K (time to expiry 0.4)")
        ax.set_ylim(-3, 55)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("option price (\$, \$100 strike)")
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle("American put: price curve through the exercise region", y=1.02)
    _save(fig, out, "fig10_price_slice")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/baselines")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--skip-showcase", action="store_true")
    ap.add_argument("--fast", action="store_true", help="tiny PINN for a quick look; not for reporting")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    part1(Path(args.results), out)
    if not args.skip_showcase:
        part2(out, args.fast)


if __name__ == "__main__":
    main()
