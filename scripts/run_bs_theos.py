"""Black-Scholes benchmark theos (flat and smile) for every symbol-date in puts_clean.csv.
Writes data/dolthub/slices/bs_theos.csv (not committed). Run from the repo root:
    PYTHONPATH=src python scripts/run_bs_theos.py --workers 6
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from ivuq.trading.bs_theo import bs_theos_for_day


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--src", default="data/dolthub/slices/puts_clean.csv")
    ap.add_argument("--out", default="data/dolthub/slices/bs_theos.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.src, parse_dates=["date", "expiry"])
    groups = [g for _, g in d.groupby(["symbol", "date"])]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(bs_theos_for_day, groups, chunksize=4))
    out = pd.concat(res)
    out[["date", "symbol", "expiry", "strike", "own_iv", "theo_flat", "theo_smile"]].to_csv(a.out, index=False)
    print(out[["own_iv", "theo_flat", "theo_smile"]].notna().mean())
    for c in ("theo_flat", "theo_smile"):
        e = (out[c] - out["mid"]).abs()
        print(c, "MAE $%.4f" % e.mean(), "median $%.4f" % e.median())


if __name__ == "__main__":
    main()
