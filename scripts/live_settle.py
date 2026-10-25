"""Settle the live signals against the snapshots taken since: the forward test ledger.

    PYTHONPATH=src python scripts/live_settle.py --tags pinn cpl1000 fd --horizon 3

Reads data/live/snapshots/*.csv and data/live/signals/<date>_<tag>.csv, writes
data/live/ledger/<tag>_h<H>_lag<lag>.csv (settled trades) and prints, per tag, the number settled and pending,
total and mean net P&L, hit rate. Nothing here is a backtest: every number comes from quotes that did not exist when
the signal was written.
"""
import argparse
from pathlib import Path

import pandas as pd

from ivuq.data.sources.yf_snapshot import SNAP_DIR, load_snapshots
from ivuq.trading.backtest import summarise
from ivuq.trading.live import settle

SIG = Path("data/live/signals")
LED = Path("data/live/ledger")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["pinn", "cpl1000", "fd"])
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--lags", nargs="+", type=int, default=[0, 1])
    a = ap.parse_args()
    LED.mkdir(parents=True, exist_ok=True)
    snaps = load_snapshots(SNAP_DIR)
    div = pd.read_csv("data/dolthub/slices/raw/dividend.csv", parse_dates=["ex_date"])
    print("snapshot dates:", [str(d.date()) for d in sorted(snaps["date"].unique())])
    for tag in a.tags:
        fs = sorted(SIG.glob(f"*_{tag}.csv"))
        if not fs:
            continue
        sig = pd.concat([pd.read_csv(f, parse_dates=["date", "expiry"]) for f in fs], ignore_index=True)
        for lag in a.lags:
            led, pend = settle(snaps, sig, a.horizon, lag, div)
            led.to_csv(LED / f"{tag}_h{a.horizon}_lag{lag}.csv", index=False)
            s = summarise(led) if len(led) else {"n_trades": 0}
            tail = f"total ${s['total_net']:.0f} mean ${s['mean_net']:.1f} hit {s['hit_rate']:.2f}" if s.get("n_trades") else ""
            print(f"{tag} lag{lag}: settled {s.get('n_trades', 0)}, pending {len(pend)} {tail}")


if __name__ == "__main__":
    main()
