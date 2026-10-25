"""Build every table of the Heston paper from the result files, so the text and the numbers cannot drift.

    PYTHONPATH=src python paper_heston/make_tables.py

Reads results/heston_trading/*.csv|json and data/dolthub/slices/{theo_*,bs_theos,puts_clean}.csv (the latter
is not committed: raw market data stays local) and writes paper_heston/tables/*.tex plus numbers.tex with the
scalar facts quoted in the text.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ivuq.pinn.heston_boxes import box_for
from ivuq.trading.study import KEY, SLICES

R = Path("results/heston_trading")
OUT = Path("paper_heston/tables")
OUT.mkdir(parents=True, exist_ok=True)
SYMS = ["SPY", "AAPL", "NVDA"]
LAB = {"pinn": "Heston single PINN", "cpl1000": "Heston coupled PINN", "fd": "Heston exact solver",
       "flat": "BS flat vol", "smile": "BS smile"}


def tex_table(name, header, rows, align, caption, label, note=None):
    lines = [r"\begin{table}[t]", r"\centering", r"\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [f"\\par\\smallskip\\footnotesize {note}"]
    lines += [r"\end{table}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def money(v):
    return f"{v:,.0f}".replace("-", "$-$") if np.isfinite(v) else "--"


def f(v, d=2):
    return f"{v:.{d}f}".replace("-", "$-$") if np.isfinite(v) else "--"


nums = {}

# ---- network accuracy against fresh exact-solver solves
rows = []
order = [("phys_pilot", "SPY", "Plain network, physics only"), ("anch_pilot", "SPY", "Plain network, anchored $w{=}5$"),
         ("cv_phys", "SPY", "Control variate, physics only"), ("cv_anch", "SPY", "Control variate, anchored $w{=}5$"),
         ("cv_anch100", "SPY", "Control variate, anchored $w{=}100$"), ("cv_anch1000", "SPY", "Control variate, anchored $w{=}1000$"),
         ("cpl1000", "SPY", "Coupled, anchored $w{=}1000$"), ("cv_anch1000", "AAPL", "Control variate, anchored $w{=}1000$"),
         ("cpl1000", "AAPL", "Coupled, anchored $w{=}1000$"), ("cv_anch1000", "NVDA", "Control variate, anchored $w{=}1000$"),
         ("cpl1000", "NVDA", "Coupled, anchored $w{=}1000$")]
for tag, sym, lab in order:
    j = json.load(open(R / f"cond_eval_{sym}_{tag}.json"))
    rows.append([sym, lab, f(j["mae"], 3), f(j["p95"], 3), f(j["mae_atm"], 3), f(j["bias"], 3)])
tex_table("accuracy", ["Symbol", "Network", "MAE", "95th pct.", "ATM MAE", "Bias"], rows, "llrrrr",
          "Price error of the parameter-conditioned networks against fresh exact-solver solves at random parameter sets "
          "(8\\,000 points, dollars per \\$100 of strike).", "tab:accuracy")

# ---- exact solver validation
v = pd.read_csv(R / "fd_validation.csv")
eu = v[v.kind == "european"]
nums["fdEuMae"] = f(eu.err.abs().mean() * 100, 2)
nums["fdEuMax"] = f(eu.err.abs().max() * 100, 2)
nums["fdEuP95"] = f(eu.err.abs().quantile(0.95) * 100, 2)
nums["fdEuN"] = str(len(eu))

# ---- pricing error against mid, by model and symbol
puts = pd.read_csv(SLICES / "puts_clean.csv", parse_dates=["date", "expiry"])
frames = {}
for tag in ("pinn", "cpl1000", "fd"):
    parts = [pd.read_csv(SLICES / f"theo_{s}_{tag}_full.csv", parse_dates=["date", "expiry"]) for s in SYMS]
    frames[tag] = pd.concat(parts)
bs = pd.read_csv(SLICES / "bs_theos.csv", parse_dates=["date", "expiry"])
base = frames["fd"][KEY + ["mid", "spread"]]
rows = []
for tag, col, df in [("fd", "theo", frames["fd"]), ("pinn", "theo", frames["pinn"]), ("cpl1000", "theo", frames["cpl1000"]),
                     ("smile", "theo_smile", bs.merge(base, on=KEY)), ("flat", "theo_flat", bs.merge(base, on=KEY))]:
    for s in SYMS:
        d = df[df.symbol == s]
        e = (d[col] - d["mid"]).abs()
        rows.append([LAB[tag], s, f(e.mean(), 3), f(e.median(), 3), f((e <= d["spread"] / 2).mean() * 100, 0) + r"\%"])
tex_table("fit", ["Theo", "Symbol", "MAE vs mid", "Median", "Within half spread"], rows, "llrrr",
          "Distance of each theo from the quoted mid (dollars), all quotes, daily walk-forward fits. The smile model has about twelve "
          "free numbers per day against five for Heston, so it fits better in sample.", "tab:fit")

# ---- parameters at the box edge
rows = []
for tag, wide in (("pinn", False), ("fd", True)):
    for s in SYMS:
        d = frames[tag][frames[tag].symbol == s].drop_duplicates("date")
        b = box_for(s, float(d["r"].median()), wide=wide)
        def edge(col, lim, up):
            span = lim[1] - lim[0]
            return float(((d[col] >= lim[1] - 1e-3 * span) if up else (d[col] <= lim[0] + 1e-3 * span)).mean())
        rows.append([LAB[tag], s, f(edge("kappa", b.kappa, True) * 100, 0) + r"\%", f(edge("xi", b.xi, True) * 100, 0) + r"\%",
                     f(edge("rho", b.rho, False) * 100, 0) + r"\%", f(d["fit_rmse"].median(), 2)])
tex_table("edge", ["Pricer", "Symbol", r"$\kappa$ at upper bound", r"$\xi$ at upper bound", r"$\rho$ at lower bound", "Median fit RMS"],
          rows, "llrrrr", "Share of daily calibrations with a parameter on the edge of its box, and the median fit residual in units of the quote "
          "error scale (a value of 1 means residuals the size of the half spread).", "tab:edge")

# ---- coverage of the conformal band
rows = []
for tag, var in (("pinn", "h_full"), ("cpl1000", "h_full"), ("fd", "h_full"), ("fd", "flat"), ("fd", "smile")):
    d = pd.read_csv(R / f"conformal_diagnostics_{tag}.csv")
    d = d[d.variant == var]
    d["miss"] = d.miss_rate * d.n
    r = []
    for band in ("conformal", "conformal_aci"):
        g = d[d.band == band]
        r.append(f(g.miss.sum() / g.n.sum() * 100, 1) + r"\%")
    sym = d[d.band == "conformal"].groupby("symbol").apply(lambda x: x.miss.sum() / x.n.sum())
    rows.append([LAB[var if var in ("flat", "smile") else tag]] + r + [f(sym.min() * 100, 1) + "--" + f(sym.max() * 100, 1) + r"\%", f"{int(d[d.band == 'conformal'].n.sum()):,}"])
tex_table("coverage", ["Theo", "Fixed $\\alpha$", "Adaptive $\\alpha$", "By symbol", "Scored quotes"], rows, "lrrrr",
          "Realised miss rate of the past-only rolling Mondrian band at target $\\alpha=10\\%$.", "tab:coverage")

# ---- in-sample backtest, horizon 3
rows = []
for tag, var in (("pinn", "h_full"), ("cpl1000", "h_full"), ("fd", "h_full"), ("pinn", "h_slow"), ("fd", "flat"), ("fd", "smile")):
    d = pd.read_csv(R / f"backtest_summary_{tag}.csv")
    rc = pd.read_csv(R / f"random_control_{tag}.csv")
    for lag in (0, 1):
        c = d[(d.variant == var) & (d.horizon == 3) & (d.rule == "conformal") & (d.lag == lag)].iloc[0]
        s = d[(d.variant == var) & (d.horizon == 3) & (d.rule == "spread") & (d.lag == lag)]
        r_ = rc[(rc.variant == var) & (rc.lag == lag)]
        lab = LAB[var if var in ("flat", "smile") else tag] + (", slow" if var == "h_slow" else "")
        rows.append([lab, "same" if lag == 0 else "next", int(c.n_trades), f(c.mean_net, 1), money(c.total_net), f(c.sharpe, 2),
                     money(s.total_net.iloc[0]) if len(s) else "--", money(r_.total_net_mean.iloc[0]) if len(r_) else "--"])
tex_table("insample", ["Theo", "Fill", "Trades", "Per trade", "Net P\\&L", "Sharpe", "No band", "Random"], rows, "llrrrrrr",
          "In-sample backtest, hold for three trading days, conformal band at $\\alpha=10\\%$. Net P\\&L in dollars after the spread, "
          "commissions and hedging cost. \\emph{No band} trades every quote that clears the spread; \\emph{Random} averages 30 random-signal runs "
          "with the trade counts of the band rule.", "tab:insample")

# ---- out of sample
rows = []
for tag, var in (("pinn", "h_full"), ("cpl1000", "h_full"), ("fd", "h_full"), ("pinn", "h_slow"), ("fd", "flat"), ("fd", "smile")):
    d = pd.read_csv(R / f"oos_summary_{tag}.csv")
    for lag in (0, 1):
        r = d[(d.variant == var) & (d.lag == lag)]
        if r.empty:
            continue
        r = r.iloc[0]
        lo, hi = json.loads(r.ci_mean_daily_pnl_ci)
        lab = LAB[var if var in ("flat", "smile") else tag] + (", slow" if var == "h_slow" else "")
        rows.append([lab, "same" if lag == 0 else "next", f"{r.alpha:.2f}/{int(r.horizon)}", int(r.n_trades), f(r.mean_net, 1), money(r.total_net),
                     f(r.sharpe, 2), f"[{money(lo)}, {money(hi)}]", money(r.control_total_mean)])
tex_table("oos", ["Theo", "Fill", r"$\alpha$/hold", "Trades", "Per trade", "Net P\&L", "Sharpe", "95\% CI, mean daily P\&L", "Random"], rows, "llrrrrrcr",
          "Out-of-sample test. The band level and holding period were chosen on the development dates by Sharpe ratio and frozen; "
          "the table is the test period only. The interval is a block bootstrap over days.", "tab:oos")

# ---- where the same-close profit comes from
rows = []
for tag, var in (("pinn", "h_full"), ("cpl1000", "h_full"), ("fd", "flat")):
    for lag in (0, 1):
        t = pd.read_csv(R / f"oos_trades_{tag}_{var}_lag{lag}.csv")
        t = t[t.status == "ok"]
        for s in SYMS:
            x = t[t.symbol == s]
            if x.empty:
                continue
            rows.append([LAB[var if var in ("flat", "smile") else tag], "same" if lag == 0 else "next", s, len(x), f(x.edge.mean(), 2),
                         f(x.opt_pnl.mean(), 0), f(x.hedge_pnl.mean(), 0), f(x.net_pnl.mean(), 0)])
tex_table("decomp", ["Theo", "Fill", "Symbol", "Trades", "Edge at entry", "Option", "Hedge", "Net per trade"], rows, "lllrrrrr",
          "Out-of-sample trades by underlying. Edge at entry is per share after crossing the spread; the other columns are dollars per contract.",
          "tab:decomp")

# ---- scalars for the text
q = puts
nums["nQuotes"] = f"{len(q):,}".replace(",", "{,}")
nums["nDates"] = str(q.date.nunique())
nums["firstDate"] = str(q.date.min().date())
nums["lastDate"] = str(q.date.max().date())
for s in SYMS:
    d = q[q.symbol == s]
    nums[f"n{s}"] = f"{len(d):,}".replace(",", "{,}")
    nums[f"spr{s}"] = f((d["spread"] / d["mid"]).median() * 100, 1)
(OUT.parent / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in nums.items()) + "\n", encoding="utf-8")
print(sorted(nums.items()))
