"""Live snapshot cleaning, parity spot, signal-to-trade conversion and settlement on synthetic data."""
import numpy as np
import pandas as pd

from ivuq.data.sources.yf_snapshot import clean_snapshot, parity_spot
from ivuq.pricing import black_scholes as bs
from ivuq.reporting.analysis import SPLIT, ablation, block_bootstrap_ci
from ivuq.trading.live import settle, signals_to_trades

R = 0.04
EXPIRIES = [pd.Timestamp("2026-11-13"), pd.Timestamp("2026-11-27")]
NO_DIV = pd.DataFrame({"ex_date": pd.to_datetime([]), "amount": [], "act_symbol": []})


def _chain(kind, spot, date, vol=0.25):
    d = pd.Timestamp(date)
    rows = []
    for ex in EXPIRIES:
        T = (ex - d).days / 365.0
        for K in np.arange(90, 111, 2.5):
            p = bs.price(spot, K, T, np.log1p(R), 0.0, vol, kind)
            rows.append(dict(strike=K, bid=round(p - 0.02, 2), ask=round(p + 0.02, 2), impliedVolatility=vol, lastPrice=p,
                             volume=10, openInterest=100, expiry=ex))
    return pd.DataFrame(rows)


def _raw(spot=100.0, date="2026-10-23"):
    raw = _chain("put", spot, date)
    raw["symbol"], raw["date"], raw["spot_close"], raw["spot"] = "TST", pd.Timestamp(date), spot, spot
    return raw


def _ust(date="2026-10-23", y=4.0):
    return pd.DataFrame({"date": [pd.Timestamp(date)], "1_month": [y], "2_month": [y], "3_month": [y], "6_month": [y], "1_year": [y]})


def _snap(date, spot=100.0):
    return clean_snapshot(_raw(spot, date), NO_DIV, _ust(date))


def test_parity_spot_recovers_quote_time_spot():
    r_of = lambda T: np.log1p(R)
    qd = pd.Timestamp("2026-10-23")
    s = parity_spot(_chain("put", 100.0, qd), _chain("call", 100.0, qd), 100.0, r_of, 0.0, qd)
    assert abs(s - 100.0) < 0.2
    # quotes struck after the close at a higher spot: the parity spot follows the quotes, not the stale close
    s2 = parity_spot(_chain("put", 101.0, qd), _chain("call", 101.0, qd), 100.0, r_of, 0.0, qd)
    assert 100.8 < s2 < 101.2


def test_clean_snapshot_schema_and_filters():
    raw = _raw()
    bad = raw.iloc[[0]].copy()
    bad["bid"] = 0.0
    c = clean_snapshot(pd.concat([raw, bad], ignore_index=True), NO_DIV, _ust())
    assert (c["bid"] > 0).all() and (c["ask"] > c["bid"]).all()
    assert c["vendor_iv"].between(0.2, 0.3).all()
    assert (c["delta"] < 0).all()
    for col in ("date", "symbol", "expiry", "strike", "spot", "bid", "ask", "mid", "spread", "dte", "tau", "m", "r", "q"):
        assert col in c


def test_settle_buy_that_reverts_makes_money_and_pending_listed():
    s0, s1, s2, s3 = (_snap(d) for d in ("2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28"))
    k = s0[s0.expiry == EXPIRIES[1]].iloc[4]
    sig = s0.copy()
    sig["theo"], sig["scale"], sig["bucket"], sig["excess"], sig["action"] = sig["mid"], 0.05, 2, 0.0, "HOLD"
    i = sig.index[(sig.strike == k.strike) & (sig.expiry == k.expiry)][0]
    sig.loc[i, ["theo", "excess", "action"]] = [k.mid + 1.0, 5.0, "BUY"]
    j = s3.index[(s3.strike == k.strike) & (s3.expiry == k.expiry)][0]
    s3.loc[j, ["bid", "ask"]] = [k.mid + 0.95, k.mid + 1.05]
    ledger, pending = settle(pd.concat([s0, s1, s2, s3], ignore_index=True), sig, horizon=3, lag=0)
    assert len(ledger) == 1 and len(pending) == 0
    assert ledger.iloc[0]["opt_pnl"] > 50
    ledger2, pending2 = settle(pd.concat([s0, s1], ignore_index=True), sig, horizon=3, lag=0)
    assert len(ledger2) == 0 and len(pending2) == 1


def test_signals_to_trades_keeps_best_per_day():
    n = 15
    sig = pd.DataFrame(dict(date=pd.Timestamp("2026-10-23"), symbol="TST", expiry=pd.Timestamp("2026-12-01"),
                            strike=np.arange(n) + 90.0, bid=1.0, ask=1.1, theo=2.0, scale=0.05, bucket=2,
                            excess=np.arange(n, dtype=float), action="BUY"))
    t = signals_to_trades(sig, max_per_day=10)
    assert len(t) == 10 and t["excess"].min() == 5.0 and (t["side"] == 1).all()


def test_block_bootstrap_and_ablation_shapes():
    rng = np.random.default_rng(0)
    lo, hi = block_bootstrap_ci(rng.normal(5, 1, 80))
    assert 4 < lo < hi < 6
    n = 40
    tr = pd.DataFrame(dict(model="pinn", rule="conformal", lag=0, date=pd.Timestamp("2026-06-01") + pd.to_timedelta(np.arange(n), "D"),
                           exit_date=pd.Timestamp("2026-06-04") + pd.to_timedelta(np.arange(n), "D"), net_pnl=rng.normal(10, 20, n)))
    tr["period"] = np.where(tr["date"] < SPLIT, "development", "test")
    ab = ablation(tr)
    assert set(ab["period"]) == {"test"} and ab.iloc[0]["n"] == n


def test_lse_occ_symbol_and_key_loading(tmp_path, monkeypatch):
    from ivuq.data.sources.lse import load_key, occ_symbol
    assert occ_symbol("AAPL", "2026-11-20", "put", 300.0) == "AAPL  261120P00300000"
    assert occ_symbol("nvda", "2026-12-18", "C", 212.5) == "NVDA  261218C00212500"
    monkeypatch.delenv("LSE_API_KEY", raising=False)
    f = tmp_path / ".env"
    f.write_text('OTHER=1\nLSE_API_KEY="abc123"\n', encoding="utf-8")
    assert load_key(f) == "abc123"
    monkeypatch.setenv("LSE_API_KEY", "fromenv")
    assert load_key(f) == "fromenv"
