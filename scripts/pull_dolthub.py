"""Pull the DoltHub window for the Heston trading study and write the cleaned
put quotes. Resumable: dates already on disk are skipped, so rerunning extends
the history. Run from the repo root.

    PYTHONPATH=src python scripts/pull_dolthub.py --start 2025-10-16 --end 2026-10-21
"""
import argparse
import json
from pathlib import Path

from ivuq.data.sources.dolthub import DEFAULT_SYMBOLS, clean_puts, load_raw, pull_raw

ap = argparse.ArgumentParser()
ap.add_argument("--start", default="2025-10-16")
ap.add_argument("--end", default="2026-10-21")
ap.add_argument("--raw", default="data/dolthub/slices/raw")
ap.add_argument("--out", default="data/dolthub/slices/puts_clean.csv")
args = ap.parse_args()

pull_raw(args.start, args.end, args.raw, DEFAULT_SYMBOLS)
puts = clean_puts(load_raw(args.raw))
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
puts.to_csv(args.out, index=False)
print(json.dumps(puts.attrs["drop_log"], indent=1))
print(puts.groupby("symbol").agg(rows=("mid", "size"), days=("date", "nunique")))
