"""Read-only data-quality checks on the DoltHub option_chain slice (AAPL, NVDA, SPY).
Inputs : slices/option_chain_AAPL_NVDA_SPY.csv (exported from local dolt clone),
         aux/ohlcv_*.csv, aux/dividend_*.csv, aux/us_treasury.csv (exported from
         local clones of post-no-preference/stocks and post-no-preference/rates).
Outputs: qc/*.csv and qc/summary.json. Nothing in paper_ieee/ or results/ is touched."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
OUT = HERE / "qc"; OUT.mkdir(exist_ok=True)
oc = pd.read_csv(HERE / "slices/option_chain_AAPL_NVDA_SPY.csv", parse_dates=["date", "expiration"])
px = pd.read_csv(HERE / "aux/ohlcv_AAPL_NVDA_SPY.csv", parse_dates=["date"])
ust = pd.read_csv(HERE / "aux/us_treasury.csv", parse_dates=["date"])
summary = {"rows_total": int(len(oc))}

# ---- coverage per symbol
cov = []
for s, g in oc.groupby("act_symbol"):
    dates = g["date"].drop_duplicates().sort_values()
    per_day = g.groupby("date").agg(n=("strike", "size"), n_exp=("expiration", "nunique"),
                                    n_strk=("strike", "nunique"))
    dte = (g["expiration"] - g["date"]).dt.days
    trade_days = px.loc[(px.act_symbol == s), "date"]
    td = trade_days[(trade_days >= dates.min()) & (trade_days <= dates.max())]
    cov.append(dict(symbol=s, rows=len(g), dates=len(dates), first=str(dates.min().date()),
                    last=str(dates.max().date()),
                    trading_days_in_span=int(td.nunique()),
                    weekend_dates=int((dates.dt.dayofweek >= 5).sum()),
                    median_rows_per_day=float(per_day.n.median()),
                    median_rows_per_day_2026=float(per_day.loc[per_day.index.year == 2026, "n"].median()),
                    median_expiries_per_day=float(per_day.n_exp.median()),
                    median_strikes_per_day=float(per_day.n_strk.median()),
                    max_dte=int(dte.max()), median_dte=float(dte.median())))
    yr = g.groupby(g.date.dt.year)["date"].nunique().rename("dates").reset_index()
    yr.insert(0, "symbol", s); yr.to_csv(OUT / f"dates_per_year_{s}.csv", index=False)
    gaps = dates.diff().dt.days
    big = pd.DataFrame({"date": dates, "gap_days": gaps}).query("gap_days > 5")
    big.to_csv(OUT / f"gaps_over_5d_{s}.csv", index=False)
pd.DataFrame(cov).to_csv(OUT / "coverage.csv", index=False)

# ---- quote quality
q = []
for s, g in oc.groupby("act_symbol"):
    q.append(dict(symbol=s,
                  null_bid=int(g.bid.isna().sum()), null_ask=int(g.ask.isna().sum()),
                  null_vol=int(g.vol.isna().sum()),
                  zero_bid_frac=float((g.bid == 0).mean()), zero_ask_frac=float((g.ask == 0).mean()),
                  crossed_frac=float((g.bid > g.ask).mean()), locked_frac=float(((g.bid == g.ask) & (g.ask > 0)).mean()),
                  rel_spread_median=float(((g.ask - g.bid) / ((g.ask + g.bid) / 2)).replace([np.inf, -np.inf], np.nan).median()),
                  vol_min=float(g.vol.min()), vol_max=float(g.vol.max()),
                  vol_at_cap_frac=float((g.vol >= 9.999).mean()),
                  delta_out_of_range=int(((g.delta.abs()) > 1).sum())))
# stale: same contract, consecutive recorded dates, identical bid & ask (bid>0)
oc_sorted = oc.sort_values(["act_symbol", "expiration", "strike", "call_put", "date"])
key = ["act_symbol", "expiration", "strike", "call_put"]
prev = oc_sorted.groupby(key)[["bid", "ask"]].shift()
same = (oc_sorted.bid == prev.bid) & (oc_sorted.ask == prev.ask) & (oc_sorted.bid > 0) & prev.bid.notna()
stale = same.groupby(oc_sorted.act_symbol).mean()
for d in q: d["unchanged_quote_vs_prev_record_frac"] = float(stale[d["symbol"]])
pd.DataFrame(q).to_csv(OUT / "quote_quality.csv", index=False)

# ---- spot join coverage (close of same date)
m = oc.merge(px[["date", "act_symbol", "close"]], on=["date", "act_symbol"], how="left")
# weekend-stamped dates (2019) carry the prior Friday's chain; match them backward up to 3 days
d_u = oc[["act_symbol", "date"]].drop_duplicates().sort_values("date")
asof = pd.merge_asof(d_u, px[["date", "act_symbol", "close"]].sort_values("date").rename(columns={"date": "px_date"}),
                     left_on="date", right_on="px_date", by="act_symbol", direction="backward", tolerance=pd.Timedelta("3D"))
summary["spot_asof_missing_dates"] = asof.groupby("act_symbol")["close"].apply(lambda x: int(x.isna().sum())).to_dict()
summary["spot_join_missing_frac"] = m.groupby("act_symbol")["close"].apply(lambda x: float(x.isna().mean())).to_dict()
summary["spot_missing_dates_sample"] = {s: [str(d.date()) for d in g.loc[g.close.isna(), "date"].drop_duplicates().head(10)]
                                        for s, g in m.groupby("act_symbol")}
# ---- rate join coverage
r = oc[["date"]].drop_duplicates().merge(ust[["date", "3_month"]], on="date", how="left")
summary["rate_missing_dates"] = int(r["3_month"].isna().sum()); summary["option_dates"] = int(len(r))
json.dump(summary, open(OUT / "summary.json", "w"), indent=2, default=str)
print(pd.DataFrame(cov).to_string()); print(pd.DataFrame(q).T.to_string()); print(json.dumps(summary, indent=1, default=str))
