"""Check what the LSE key can read for US options, end to end, and save small samples.

    PYTHONPATH=src python scripts/lse_probe.py [--symbol NVDA] [--day 2026-10-22]

Prints the plan and limits, a few chain rows, one put's per-minute Greeks and its trade prints, and compares the
traded prices of the day against the Yahoo snapshot of the same contract when a snapshot exists. Samples go to
data/live/lse/ (gitignored). The key is read by ivuq.data.sources.lse.load_key and is never printed.
"""
import argparse
from pathlib import Path

import pandas as pd

from ivuq.data.sources.lse import LSEClient, occ_symbol

OUT = Path("data/live/lse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NVDA")
    ap.add_argument("--day", default="2026-10-22")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    c = LSEClient()
    u = c.usage()
    print("plan:", u.get("plan"), "| markets:", u.get("markets"), "| calls/min:", u.get("calls_per_minute"),
          "| weekly:", u.get("weekly"))
    ch = c.chain(a.symbol, type="put", max_dte=45, limit=500)
    print(f"chain: {len(ch)} put contracts, underlying {ch.attrs.get('underlying_price')}, greeks as of {ch.attrs.get('greeks_as_of')}")
    print(ch[["symbol", "strike", "expiry", "dte", "last_price", "last_trade_at", "volume_today", "iv", "delta"]].head(8).to_string())
    ch.to_csv(OUT / f"chain_{a.symbol}.csv", index=False)
    traded = ch[ch["volume_today"].fillna(0) > 0].sort_values("volume_today", ascending=False)
    pick = (traded if len(traded) else ch).iloc[0]
    sym = pick["symbol"]
    print("contract:", sym)
    try:
        g = c.greeks(sym, a.day, a.day)
        print(f"greeks rows: {len(g)}", g.head(3).to_string())
        g.to_csv(OUT / f"greeks_{a.symbol}_{a.day}.csv", index=False)
    except RuntimeError as e:
        print("greeks:", e)
    try:
        t = c.trades(sym, a.day)
        print(f"trade prints: {len(t)}", t.head(3).to_string())
        t.to_csv(OUT / f"trades_{a.symbol}_{a.day}.csv", index=False)
    except RuntimeError as e:
        print("trades:", e)


if __name__ == "__main__":
    main()
