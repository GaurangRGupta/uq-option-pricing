"""Extra read-only checks: sentinels, moneyness, missing-spot dates, expiry pattern,
NVDA split handling, 2026-08-13 comparison with the Yahoo snapshot, arbitrage checker."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = HERE / "qc"
oc = pd.read_csv(HERE / "slices/option_chain_AAPL_NVDA_SPY.csv", parse_dates=["date", "expiration"])
px = pd.read_csv(HERE / "aux/ohlcv_AAPL_NVDA_SPY.csv", parse_dates=["date"])
ust = pd.read_csv(HERE / "aux/us_treasury.csv", parse_dates=["date"])
res = {}
res["vol_le_0_by_symbol"] = oc[oc.vol <= 0].groupby("act_symbol").size().to_dict()
res["vol_eq_minus1_by_symbol"] = oc[oc.vol == -1].groupby("act_symbol").size().to_dict()
m = oc.merge(px[["date", "act_symbol", "close"]], on=["date", "act_symbol"], how="left")
miss = m[m.close.isna()][["act_symbol", "date"]].drop_duplicates()
miss = miss[miss.date.dt.dayofweek < 5]
res["weekday_dates_without_same_day_close"] = {s: [str(d.date()) for d in g.date] for s, g in miss.groupby("act_symbol")}
mm = m.dropna(subset=["close"]).copy(); mm["kS"] = mm.strike / mm.close
res["K_over_S_quantiles"] = {s: g.kS.quantile([0, .05, .25, .5, .75, .95, 1]).round(3).to_dict() for s, g in mm.groupby("act_symbol")}
d26 = oc[oc.date.dt.year == 2026]
res["expiries_per_day_2026_dist"] = d26.groupby(["act_symbol", "date"]).expiration.nunique().value_counts().sort_index().to_dict()
res["dte_quantiles_2026"] = (d26.expiration - d26.date).dt.days.quantile([0, .25, .5, .75, 1]).to_dict()
res["expiration_weekday_counts"] = oc.drop_duplicates(["act_symbol", "expiration"]).expiration.dt.day_name().value_counts().to_dict()
nv = oc[(oc.act_symbol == "NVDA") & oc.date.isin(pd.to_datetime(["2024-06-07", "2024-06-10"]))]
res["nvda_strike_range_around_2024_split"] = nv.groupby("date").strike.agg(["min", "max"]).astype(float).rename(index=str).to_dict()
# ---- 2026-08-13 vs Yahoo snapshot
cmp = {}
for s in ["AAPL", "NVDA"]:
    y = pd.read_csv(ROOT / f"data/raw/{s}_2026-08-13.csv", parse_dates=["expiry_date", "quote_date"])
    d = oc[(oc.act_symbol == s) & (oc.date == "2026-08-13")].copy()
    d["option_type"] = d.call_put.str.lower(); d["mid"] = (d.bid + d.ask) / 2
    j = d.merge(y, left_on=["expiration", "strike", "option_type"], right_on=["expiry_date", "strike", "option_type"], suffixes=("_d", "_y"))
    jj = j[(j.bid_d > 0) & (j.bid_y > 0)]
    # implied spot from parity, nearest expiry, 5 strikes closest to ATM (r from snapshot, ignore q)
    e0 = d.expiration.min(); dd = d[d.expiration == e0]
    piv = dd.pivot_table(index="strike", columns="option_type", values="mid").dropna()
    T = (e0 - pd.Timestamp("2026-08-13")).days / 365; r = float(y.risk_free_rate.iloc[0])
    piv["S_impl"] = piv["call"] - piv["put"] + piv.index.values * np.exp(-r * T)
    close = px[(px.act_symbol == s) & px.date.isin(pd.to_datetime(["2026-08-12", "2026-08-13"]))].set_index("date").close
    atm = piv.iloc[(piv["call"] - piv["put"]).abs().argsort()[:5]]
    cmp[s] = dict(dolthub_rows=len(d), yahoo_rows=len(y), dolthub_expiries=[str(x.date()) for x in sorted(d.expiration.unique())],
                  yahoo_expiries=int(y.expiry_date.nunique()), matched_contracts=len(j), matched_both_bid_pos=len(jj),
                  median_abs_mid_diff=float((jj.mid - (jj.bid_y + jj.ask_y) / 2).abs().median()) if len(jj) else None,
                  median_abs_iv_diff_vs_vendor=float((jj.vol - jj.vendor_implied_volatility).abs().median()) if len(jj) else None,
                  yahoo_underlying_price=float(y.underlying_price.iloc[0]),
                  close_2026_08_12=float(close.iloc[0]), close_2026_08_13=float(close.iloc[1]),
                  parity_implied_spot_nearest_expiry_median=float(atm.S_impl.median()), nearest_expiry=str(e0.date()))
res["compare_2026_08_13"] = cmp
# ---- arbitrage checker on DoltHub data mapped to ivuq schema (SPY treated as American ETF)
from ivuq.arbitrage.checker import check_arbitrage
div = pd.read_csv(HERE / "aux/dividend_AAPL_NVDA_SPY.csv", parse_dates=["ex_date"])
arb = []
for s in ["AAPL", "NVDA", "SPY"]:
    for dt in ["2024-03-13", "2025-06-11", "2026-08-13", "2026-10-09"]:
        dt = pd.Timestamp(dt)
        d = oc[(oc.act_symbol == s) & (oc.date == dt) & (oc.bid > 0)].copy()
        if d.empty: arb.append(dict(symbol=s, date=str(dt.date()), rows=0)); continue
        S = float(px[(px.act_symbol == s) & (px.date == dt)].close.iloc[0])
        r3 = float(ust[ust.date <= dt].iloc[-1]["3_month"]) / 100
        trailing = div[(div.act_symbol == s) & (div.ex_date > dt - pd.Timedelta("365D")) & (div.ex_date <= dt)].amount.sum()
        q = float(np.log1p(trailing / S))
        f = pd.DataFrame(dict(underlying=s, option_style="american", option_type=d.call_put.str.lower(),
                              strike=d.strike.astype(float), expiry_date=d.expiration, quote_date=d.date,
                              underlying_price=S, market_option_price=(d.bid + d.ask) / 2,
                              risk_free_rate=np.log1p(r3), dividend_yield=q, data_source="dolthub",
                              bid=d.bid, ask=d.ask, bid_ask_spread=d.ask - d.bid))
        rep = check_arbitrage(f)
        arb.append(dict(symbol=s, date=str(dt.date()), rows=len(f), report=str(rep).replace("\n", " | ")))
res["arbitrage_checker"] = arb
json.dump(res, open(OUT / "extra.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
