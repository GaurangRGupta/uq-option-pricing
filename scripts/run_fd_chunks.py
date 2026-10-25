"""Run the FD-calibrated Heston theos for several symbols/modes in parallel date chunks, then merge.

    PYTHONPATH=src python scripts/run_fd_chunks.py --symbols SPY AAPL NVDA --modes full slow --chunks 8 --workers 12

Each chunk is one `run_heston_theo.py --pricer fd` process over a slice of dates plus a warm-up of earlier dates
(fitted, then dropped). Chunk starts are cold, so the first rows of a chunk use a fresh multi-start fit; this
only affects parameter smoothness at chunk joins, not look-ahead (every fit still uses past and current dates only).
Merged files: data/dolthub/slices/theo_<SYM>_fd_<mode>.csv
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

SL = Path("data/dolthub/slices")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", default=["SPY", "AAPL", "NVDA"])
    ap.add_argument("--modes", nargs="+", default=["full"])
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="fd")
    ap.add_argument("--max-nfev", type=int, default=15)
    ap.add_argument("--refit-every", type=int, default=5)
    a = ap.parse_args()
    d = pd.read_csv(SL / "puts_clean.csv", parse_dates=["date"])
    jobs, merged = [], []
    for s in a.symbols:
        dates = sorted(d.loc[d["symbol"] == s, "date"].unique())
        step = -(-len(dates) // a.chunks)
        for mode in a.modes:
            files = []
            for i in range(0, len(dates), step):
                lo, hi = dates[i], dates[min(i + step, len(dates)) - 1]
                f = SL / f"theo_{s}_{a.tag}_{mode}_{lo.date()}.csv"
                files.append(f)
                if f.exists():
                    continue
                cmd = [sys.executable, "scripts/run_heston_theo.py", "--symbol", s, "--pricer", "fd", "--tag", a.tag,
                       "--mode", mode, "--start", str(lo.date()), "--end", str(hi.date()), "--warmup", "12",
                       "--max-nfev", str(a.max_nfev), "--refit-every", str(a.refit_every)]
                jobs.append(cmd)
            merged.append((s, mode, files))
    print(f"{len(jobs)} chunk jobs", flush=True)

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        print("done" if r.returncode == 0 else "FAILED", cmd[4], cmd[8], cmd[10], flush=True)
        if r.returncode:
            print(r.stderr[-800:], flush=True)
        return r.returncode

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(run, jobs))
    for s, mode, files in merged:
        parts = [pd.read_csv(f, parse_dates=["date", "expiry"]) for f in files if f.exists()]
        if parts:
            out = pd.concat(parts).drop_duplicates(["date", "symbol", "expiry", "strike"]).sort_values(["date", "expiry", "strike"])
            out.to_csv(SL / f"theo_{s}_{a.tag}_{mode}.csv", index=False)
            e = (out["theo"] - out["mid"]).abs()
            print(f"{s} {mode}: {len(out)} rows from {len(parts)} chunks; MAE ${e.mean():.4f} median ${e.median():.4f}")


if __name__ == "__main__":
    main()
