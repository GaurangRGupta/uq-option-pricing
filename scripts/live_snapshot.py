"""Take one Yahoo put-chain snapshot per symbol, saved under data/live/snapshots/<quote_date>_<symbol>.csv.

    PYTHONPATH=src python scripts/live_snapshot.py

Run once per trading day shortly after the close (the file of a date is overwritten if you run twice the same day).
Daily routine for the forward test:
    python scripts/live_snapshot.py && python scripts/live_signals.py && python scripts/live_settle.py
"""
from ivuq.data.sources.yf_snapshot import take_snapshot

if __name__ == "__main__":
    d = take_snapshot()
    print(d.groupby("symbol").agg(date=("date", "first"), quotes=("mid", "size"), spot=("spot", "first")).to_string())
