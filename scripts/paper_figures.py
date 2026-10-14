"""Publication figures for the paper (IEEE single column, 3.5 in wide, greyscale).

    python scripts/paper_figures.py

Reads the CSVs written by paper_p3_synthetic.py, paper_p1p2_conformal.py and
paper_p4_arbitrage.py; writes PDF + PNG to results/uq/paper/figures/. The
slice and residual figures also reload the seed-0 PINN and the seed-0 pool;
every number plotted is also written to a CSV next to the figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ivuq.baselines.data import MarketParams, reference_u
from ivuq.uq import split_conformal_interval

from paper_common import BUCKET_NAMES, PAPER
from paper_models import K, predictors

import os

FIG = Path(os.environ["PAPER_FIG_OUT"]) if os.environ.get("PAPER_FIG_OUT") else PAPER / "figures"
W = 3.5
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"], "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
GREYS = ["0.15", "0.45", "0.7", "0.9"]
HATCH = ["", "///", "...", "xxx"]
MLABEL = {"vanilla": "Split", "normalised_pde": "Norm. |L[u]|", "mondrian": "Mondrian", "normalised_lcp": "Norm. LCP",
          "normalised": "Normalised"}
BLABEL = ["m<0.9\nshort", "m<0.9\nlong", "ATM\nshort", "ATM\nlong", "m>1.1\nshort", "m>1.1\nlong"]


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)


def bucket_bars(summ, methods, alpha, name, ylabel, kind="cov", ylim=None, target=None):
    fig, ax = plt.subplots(figsize=(W, 2.1))
    x = np.arange(6)
    wbar = 0.8 / len(methods)
    for i, meth in enumerate(methods):
        row = summ[summ.method == meth].iloc[0]
        mean = [row[f"{kind}_b{k}_mean"] for k in range(6)]
        std = [row[f"{kind}_b{k}_std"] for k in range(6)]
        ax.bar(x - 0.4 + wbar * (i + 0.5), mean, wbar, yerr=std, color=GREYS[i], hatch=HATCH[i], edgecolor="black",
               linewidth=0.4, error_kw=dict(lw=0.5, capsize=1.2), label=MLABEL[meth])
    if target is not None:
        ax.axhline(target, color="black", ls="--", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(BLABEL)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(ncol=len(methods), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handlelength=1.2, columnspacing=0.8)
    save(fig, name)


def fig_p3():
    s = pd.read_csv(PAPER / "p3" / "p3_summary.csv")
    for alpha in (0.10, 0.05):
        sub = s[(s.experiment == "heteroscedastic") & np.isclose(s.alpha, alpha)]
        a = int(round(alpha * 100))
        bucket_bars(sub, ["vanilla", "normalised", "mondrian"], alpha, f"fig_p3_bucket_coverage_a{a:02d}",
                    "Coverage", ylim=(0.6, 1.02), target=1 - alpha)


def fig_marginal():
    s = pd.read_csv(PAPER / "p1p2" / "p1p2_summary.csv")
    s = s[s.n_cal == 800]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.0), sharey=False)
    order = [("n0", "vanilla"), ("n0", "normalised_pde"), ("n0", "mondrian"),
             ("n1", "vanilla"), ("n1", "normalised_pde"), ("n1", "mondrian"),
             ("n2", "vanilla"), ("n2", "normalised_pde"), ("n2", "mondrian"), ("n2", "normalised_lcp")]
    marks = {"vanilla": "o", "normalised_pde": "s", "mondrian": "^", "normalised_lcp": "D"}
    rows = []
    for ax, alpha in zip(axes, (0.10, 0.05)):
        for j, (mod, meth) in enumerate(order):
            r = s[(s.model == mod) & (s.method == meth) & np.isclose(s.alpha, alpha)].iloc[0]
            ax.errorbar(j, r.coverage_mean, yerr=r.coverage_std, fmt=marks[meth], color="black", mfc="white",
                        ms=3.5, lw=0.6, capsize=1.5)
            rows.append({"alpha": alpha, "model": mod, "method": meth, "coverage_mean": r.coverage_mean, "coverage_std": r.coverage_std})
        ax.axhline(1 - alpha, color="black", ls="--", lw=0.7)
        ax.set_xticks([1, 4, 7.5])
        ax.set_xticklabels(["N0", "N1", "N2"])
        for b in (2.5, 5.5):
            ax.axvline(b, color="0.8", lw=0.5)
        ax.set_title(f"alpha = {alpha:.2f}")
    axes[0].set_ylabel("Marginal coverage")
    handles = [plt.Line2D([], [], marker=marks[m], ls="", color="black", mfc="white", ms=3.5, label=MLABEL[m]) for m in marks]
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.98), frameon=False, handletextpad=0.3)
    fig.tight_layout()
    save(fig, "fig_p1p2_marginal_coverage")
    pd.DataFrame(rows).to_csv(FIG / "fig_p1p2_marginal_coverage.csv", index=False)


def fig_p2_buckets():
    s = pd.read_csv(PAPER / "p1p2" / "p1p2_summary.csv")
    meths = ["vanilla", "normalised_pde", "mondrian", "normalised_lcp"]
    for alpha in (0.10, 0.05):
        a = int(round(alpha * 100))
        sub = s[(s.model == "n2") & (s.n_cal == 800) & np.isclose(s.alpha, alpha)]
        bucket_bars(sub, meths, alpha, f"fig_p2_bucket_coverage_a{a:02d}", "Coverage", ylim=(0.4, 1.02), target=1 - alpha)
        bucket_bars(sub, meths, alpha, f"fig_p2_bucket_width_a{a:02d}", "Mean width ($)", kind="width")
    sub = s[(s.model == "n1") & (s.n_cal == 800) & np.isclose(s.alpha, 0.10)]
    bucket_bars(sub, ["vanilla", "normalised_pde", "mondrian"], 0.10, "fig_p1_n1_bucket_coverage_a10", "Coverage", ylim=(0.4, 1.02), target=0.9)


def fig_slice():
    torch.set_num_threads(4)
    seed, alpha, n_cal, tau0 = 0, 0.10, 800, 0.5
    pool = pd.read_csv(PAPER / "pools" / f"pool_seed{seed}.csv")
    perm = np.random.default_rng(200000 + 100 * seed + 0).permutation(len(pool))
    ci = perm[:n_cal]
    mc, tc, uc = pool.m.to_numpy()[ci], pool.tau.to_numpy()[ci], pool.u_am_crr500.to_numpy()[ci]
    preds = predictors("n2", seed)
    m = np.linspace(0.6, 1.4, 81)
    tau = np.full_like(m, tau0)
    ref = reference_u("american", "put", m, tau, MarketParams(), tree_steps=500)
    van = split_conformal_interval(preds["pde"], mc, tc, uc, m, tau, alpha=alpha, kappa=0.0)
    lcp = split_conformal_interval(preds["lcp"], mc, tc, uc, m, tau, alpha=alpha, scale_kappa=True)
    df = pd.DataFrame({"m": m, "tau": tau, "crr500_usd": K * ref, "pinn_usd": K * van.u_hat,
                       "vanilla_lo_usd": K * van.lo, "vanilla_hi_usd": K * van.hi,
                       "lcp_lo_usd": K * lcp.lo, "lcp_hi_usd": K * lcp.hi, "intrinsic_usd": K * np.maximum(1 - m, 0)})
    FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG / "fig_p2_interval_slice.csv", index=False)
    fig, ax = plt.subplots(figsize=(W, 2.3))
    ax.fill_between(m, K * van.lo, K * van.hi, color="0.85", lw=0, label="Split 90% band")
    ax.plot(m, K * lcp.lo, color="0.35", ls=":", lw=1.0, label="Norm. LCP 90% band")
    ax.plot(m, K * lcp.hi, color="0.35", ls=":", lw=1.0)
    ax.plot(m, K * ref, color="black", lw=1.2, label="CRR (500 steps)")
    ax.plot(m, K * van.u_hat, color="black", ls="--", lw=0.9, label="N2 PINN")
    ax.plot(m, K * np.maximum(1 - m, 0), color="0.55", lw=0.6, ls="-.", label="Intrinsic")
    ax.set_xlabel("Moneyness m = S/K")
    ax.set_ylabel("American put price ($, K = 100)")
    ax.set_xlim(0.6, 1.4)
    ax.legend(frameon=False, loc="upper right")
    save(fig, "fig_p2_interval_slice_tau050")


def fig_resid():
    torch.set_num_threads(4)
    pool = pd.read_csv(PAPER / "pools" / "pool_seed0.csv")
    preds = predictors("n2", 0)
    m, tau, u = pool.m.to_numpy(), pool.tau.to_numpy(), pool.u_am_crr500.to_numpy()
    err = K * np.abs(preds["pde"].predict(m, tau) - u)
    rp = np.abs(preds["pde"].constraint_residual(m, tau))
    rl = np.abs(preds["lcp"].constraint_residual(m, tau))
    pd.DataFrame({"m": m, "tau": tau, "abs_err_usd": err, "abs_pde_resid": rp, "abs_lcp_resid": rl}).to_csv(
        FIG / "fig_p2_residual_vs_error.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(W, 1.9), sharey=True)
    for ax, r, lab in ((axes[0], rp, "|L[u]|"), (axes[1], rl, "|min(L[u], u-h)|")):
        ax.scatter(r, err, s=1.5, color="black", alpha=0.25, lw=0, rasterized=True)
        ax.set_xscale("log")
        ax.set_xlabel(lab)
    axes[0].set_ylabel("|PINN - CRR| ($)")
    fig.tight_layout()
    save(fig, "fig_p2_residual_vs_error")


def fig_p4():
    p = PAPER / "p4" / "p4_summary.csv"
    if not p.exists():
        return
    s = pd.read_csv(p)
    s = s[(s.model == "n2") & np.isclose(s.tol, 1e-6)]
    checks = ["strike_monotonicity", "strike_convexity", "calendar_spread"]
    series = [("pinn", "point", "PINN point")] + [(m, e, f"{MLABEL[m]} {e}") for m in ("vanilla", "normalised_lcp") for e in ("lo", "hi")]
    fig, ax = plt.subplots(figsize=(W, 2.0))
    x = np.arange(len(checks))
    wbar = 0.8 / len(series)
    grey = ["0.1", "0.4", "0.6", "0.8", "0.95"]
    hatch = ["", "///", "", "...", ""]
    for i, (meth, edge, lab) in enumerate(series):
        vals = [s[(s.method == meth) & (s.edge == edge) & (s.check == c)].rate_mean.iloc[0] for c in checks]
        ax.bar(x - 0.4 + wbar * (i + 0.5), vals, wbar, color=grey[i], hatch=hatch[i], edgecolor="black", lw=0.4, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["Monotone in K", "Convex in K", "Calendar"])
    ax.set_ylabel("Violation rate")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=0.8)
    save(fig, "fig_p4_n2_arbitrage")


def main():
    fig_p3()
    fig_marginal()
    fig_p2_buckets()
    fig_slice()
    fig_resid()
    fig_p4()
    print("figures in", FIG)


if __name__ == "__main__":
    main()
