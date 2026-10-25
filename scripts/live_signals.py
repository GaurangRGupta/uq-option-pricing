"""Run the Heston theos and the past-only conformal band on live Yahoo snapshots.

For each snapshot date not yet processed, per symbol: warm up the calibration on the last `--warm` dated
DoltHub days, fit the snapshot date with the chosen pricer (tag pinn = single conditioned network, cpl1000 =
coupled network, fd = exact solver), price every quote, add the BS flat and smile comparison theos, and build the
conformal band from the previous 20 dates of the historical panel (scores |mid - theo| / c) plus any earlier live
dates. Writes data/live/signals/<date>_<tag>.csv with the columns of scripts/make_signals.py plus the vendor-free
Greeks, and prints coverage and error diagnostics.

    PYTHONPATH=src python scripts/live_signals.py --tags pinn cpl1000 fd [--date 2026-10-23]

The history (DoltHub panel) ends 2026-10-21; live dates after it are scored against that history, which is the
real test of whether the band still covers on a different quote source (Yahoo versus DoltHub).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ivuq.calibration.heston_calib import FDPricer, PINNPricer, error_scale
from ivuq.data.sources.yf_snapshot import SNAP_DIR, load_snapshots
from ivuq.pinn.heston_boxes import box_for
from ivuq.trading.bs_theo import bs_theos_for_day
from ivuq.trading.online_conformal import add_rolling_conformal, bucket_ids
from ivuq.trading.study import KEY, load_panel
from ivuq.trading.theo_pipeline import run_theo

OUT = Path("data/live/signals")
MODEL = {"pinn": "models/heston_cond/{s}_cv_anch1000.pt", "cpl1000": "models/heston_cond/{s}_cpl1000.pt"}


def load_model(path):
    blob = torch.load(path, weights_only=False)
    if blob.get("kind") == "coupled":
        from ivuq.pinn.heston_cond_coupled import HestonCoupledPINN
        return HestonCoupledPINN.load(path)
    from ivuq.pinn.heston_cond import HestonCondPINN
    return HestonCondPINN.load(path)


def theo_for(tag, sym, hist, snap, warm):
    d = pd.concat([hist, snap], ignore_index=True)
    d["scale"] = error_scale(d)
    if tag == "fd":
        box = box_for(sym, wide=True)
        pricer, final = FDPricer(box, grid="mid"), FDPricer(box, grid="fine")
    else:
        pricer, final = PINNPricer(load_model(MODEL[tag].format(s=sym))), None
    out = run_theo(pricer, d, mode="full", final_pricer=final, warmup_dates=warm, max_nfev=25)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["pinn", "cpl1000", "fd"])
    ap.add_argument("--symbols", nargs="+", default=["SPY", "AAPL", "NVDA"])
    ap.add_argument("--date", default=None)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    snaps = load_snapshots(SNAP_DIR)
    dates = sorted(snaps["date"].unique())
    if a.date:
        dates = [pd.Timestamp(a.date)]
    puts = pd.read_csv("data/dolthub/slices/puts_clean.csv", parse_dates=["date", "expiry"])
    for dt in dates:
        sn = snaps[(snaps["date"] == dt) & snaps["symbol"].isin(a.symbols)].reset_index(drop=True)
        # comparison theos on the snapshot (same fitting information as the Heston calibration)
        bsf = pd.concat([bs_theos_for_day(g) for _, g in sn.groupby("symbol")], ignore_index=True)
        for tag in a.tags:
            f = OUT / f"{dt.date()}_{tag}.csv"
            parts = []
            for sym in a.symbols:
                hist = puts[(puts["symbol"] == sym)]
                hist = hist[hist["date"].isin(sorted(hist["date"].unique())[-a.warm:])]
                cols = list(puts.columns)
                s_sym = sn[sn["symbol"] == sym][cols]
                th = theo_for(tag, sym, hist, s_sym, a.warm)
                parts.append(th)
                print(tag, sym, dt.date(), len(th), f"MAE vs mid ${(th['theo'] - th['mid']).abs().mean():.3f}",
                      f"fit_rmse {th['fit_rmse'].iloc[0]:.2f}", flush=True)
            live = pd.concat(parts, ignore_index=True)
            live = live.merge(sn[KEY + ["last", "volume", "open_interest", "yahoo_iv"]], on=KEY, how="left")
            live = live.merge(bsf[KEY + ["theo_flat", "theo_smile"]], on=KEY, how="left")
            live["scale"] = error_scale(live)
            live["bucket"] = bucket_ids(live["m"].to_numpy(), live["dte"].to_numpy())
            # band: previous dates of the historical panel (same tag) plus earlier live dates
            panel, variants = load_panel(tag, a.symbols, with_bs=False)
            h = panel[KEY + ["m", "dte", "mid", "scale", "bucket", "theo_h_full"]].rename(columns={"theo_h_full": "theo"})
            prior = [pd.read_csv(g, parse_dates=["date", "expiry"]) for g in sorted(OUT.glob(f"*_{tag}.csv"))
                     if g.name < f.name]
            lv = live[KEY + ["m", "dte", "mid", "scale", "bucket", "theo"]]
            allp = pd.concat([h] + [p[KEY + ["m", "dte", "mid", "scale", "bucket", "theo"]] for p in prior] + [lv],
                             ignore_index=True)
            pc = add_rolling_conformal(allp, "theo", alpha=a.alpha, window=a.window, gamma=0.0)
            pc = pc[pc["date"] == dt][KEY + ["q", "n_cal"]].rename(columns={"q": "cq"})
            live = live.merge(pc, on=KEY, how="left")
            live["half_width"] = live["cq"] * live["scale"]
            live["lo"], live["hi"] = live["theo"] - live["half_width"], live["theo"] + live["half_width"]
            buy, sell = live["theo"] - live["ask"], live["bid"] - live["theo"]
            live["edge"] = np.where(buy >= sell, buy, -sell)
            live["action"] = np.where(buy > live["half_width"], "BUY", np.where(sell > live["half_width"], "SELL", "HOLD"))
            live.loc[~np.isfinite(live["half_width"]), "action"] = "NO_BAND"
            gap = np.where(live["action"] == "BUY", buy, np.where(live["action"] == "SELL", sell, 0.0))
            live["excess"] = np.where(live["action"].isin(["BUY", "SELL"]), (gap - live["half_width"]) / live["scale"], 0.0)
            live["hedge_shares"] = np.where(live["action"] == "BUY", -100 * live["delta"],
                                            np.where(live["action"] == "SELL", 100 * live["delta"], 0.0))
            live["covered"] = (live["mid"] - live["theo"]).abs() <= live["half_width"]
            live.to_csv(f, index=False)
            cov = live.loc[np.isfinite(live["half_width"]), "covered"]
            print(f"{f}: n={len(live)} band coverage {cov.mean():.3f} (target {1 - a.alpha:.2f}) "
                  f"actions {live['action'].value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
