"""Daily trading signals from a Heston theo and the past-only conformal band.

One row per quote on the chosen date with: bid, ask, mid, theo, the band [theo - h, theo + h]
(h = q * c, q the conformal quantile of |mid - theo| / c from the previous --window dates, c the error
scale), the signed edge, and the action:
  BUY   if theo - ask > h        (market offers below the lower band edge, after crossing the spread)
  SELL  if bid - theo > h        (market bids above the upper band edge)
  HOLD  otherwise (NO_BAND when there is not enough history for a quantile)
plus the vendor-delta hedge in shares for one contract. Rows are ranked by (edge - h) / c.

    PYTHONPATH=src python scripts/make_signals.py --tag fd --variant h_full --alpha 0.10            # latest date
    PYTHONPATH=src python scripts/make_signals.py --tag fd --variant h_full --all                  # whole history

Writes results/heston_trading/signals_<tag>_<variant>[_<date>].csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ivuq.trading.online_conformal import add_rolling_conformal
from ivuq.trading.study import load_panel

OUT = Path("results/heston_trading")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fd")
    ap.add_argument("--variant", default="h_full")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--date", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--symbols", nargs="+", default=["SPY", "AAPL", "NVDA"])
    a = ap.parse_args()
    panel, variants = load_panel(a.tag, a.symbols)
    col = dict(variants)[a.variant]
    pc = add_rolling_conformal(panel, col, alpha=a.alpha, window=a.window, gamma=a.gamma)
    pc["theo"] = pc[col]
    pc["half_width"] = pc["q"] * pc["scale"]
    pc["lo"], pc["hi"] = pc["theo"] - pc["half_width"], pc["theo"] + pc["half_width"]
    buy, sell = pc["theo"] - pc["ask"], pc["bid"] - pc["theo"]
    pc["edge"] = np.where(buy >= sell, buy, -sell)                      # positive: market cheap versus theo
    pc["action"] = np.where(buy > pc["half_width"], "BUY", np.where(sell > pc["half_width"], "SELL", "HOLD"))
    pc.loc[~np.isfinite(pc["half_width"]), "action"] = "NO_BAND"
    gap = np.where(pc["action"] == "BUY", buy, np.where(pc["action"] == "SELL", sell, 0.0))
    pc["excess"] = (gap - pc["half_width"]) / pc["scale"]
    pc.loc[pc["action"].isin(["HOLD", "NO_BAND"]), "excess"] = 0.0
    pc["hedge_shares"] = np.where(pc["action"] == "BUY", -100 * pc["delta"], np.where(pc["action"] == "SELL", 100 * pc["delta"], 0.0))
    cols = ["date", "symbol", "expiry", "strike", "dte", "spot", "bid", "ask", "mid", "theo", "lo", "hi", "edge",
            "excess", "action", "hedge_shares", "delta", "scale", "q", "alpha_t"]
    out = pc[cols]
    OUT.mkdir(parents=True, exist_ok=True)
    if a.all:
        f = OUT / f"signals_{a.tag}_{a.variant}.csv"
        out.to_csv(f, index=False)
    else:
        d = pd.Timestamp(a.date) if a.date else out["date"].max()
        out = out[out["date"] == d].sort_values(["action", "excess"], ascending=[True, False])
        f = OUT / f"signals_{a.tag}_{a.variant}_{d.date()}.csv"
        out.to_csv(f, index=False)
    print(f, len(out))
    print(out["action"].value_counts().to_string())


if __name__ == "__main__":
    main()
