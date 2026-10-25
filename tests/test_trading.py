"""Rolling conformal band and the take-the-quote backtest, on synthetic panels."""

import numpy as np
import pandas as pd

from ivuq.trading.backtest import BacktestConfig, run_backtest, summarise
from ivuq.trading.online_conformal import add_rolling_conformal, bucket_ids


def _panel(n_days=50, n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        for i in range(n):
            mid = 5.0 + rng.normal()
            rows.append(dict(date=pd.Timestamp("2026-01-01") + pd.Timedelta(days=d), symbol="X", expiry=pd.Timestamp("2026-06-01"),
                             strike=100.0 + i, spot=100.0, bid=mid - 0.1, ask=mid + 0.1, mid=mid, spread=0.2,
                             dte=30, tau=30 / 365, m=0.9 + 0.2 * i / n, scale=0.1, theo=mid + rng.exponential(0.3) * rng.choice([-1, 1]),
                             delta=-0.5, model_delta=-0.5))
    return pd.DataFrame(rows)


def test_buckets_cover_six_cells():
    b = bucket_ids(np.array([0.9, 0.9, 1.0, 1.0, 1.1, 1.1]), np.array([10, 40, 10, 40, 10, 40]))
    assert sorted(b) == [0, 1, 2, 3, 4, 5]


def test_conformal_is_past_only_and_tracks_alpha():
    p = _panel()
    out = add_rolling_conformal(p, "theo", alpha=0.10, window=20, min_cal=60)
    first = out[out["date"] == out["date"].min()]
    assert np.isinf(first["q"]).all()                       # nothing in the past: no band, so no trade
    later = out[np.isfinite(out["q"])]
    miss = (later["score"] > later["q"]).mean()
    assert 0.05 < miss < 0.16                                # iid scores: rolling split conformal is close to alpha
    # causality: changing the last day cannot change any earlier quantile
    p2 = p.copy()
    last = p2["date"].max()
    p2.loc[p2["date"] == last, "theo"] += 100.0
    out2 = add_rolling_conformal(p2, "theo", alpha=0.10, window=20, min_cal=60)
    a, b = out[out["date"] < last]["q"].to_numpy(), out2[out2["date"] < last]["q"].to_numpy()
    assert np.array_equal(a, b)


def test_adaptive_level_reacts_to_a_regime_shift():
    p = _panel(n_days=80)
    shift = p["date"] > p["date"].min() + pd.Timedelta(days=40)
    p.loc[shift, "theo"] = p.loc[shift, "mid"] + 3.0        # model suddenly far from the market
    fixed = add_rolling_conformal(p, "theo", alpha=0.10, gamma=0.0)
    adapt = add_rolling_conformal(p, "theo", alpha=0.10, gamma=0.05)
    assert adapt["alpha_t"].min() < 0.10 - 1e-9              # level is lowered after misses, so the band widens
    assert adapt["alpha_t"].nunique() > 1 and fixed["alpha_t"].nunique() == 1


def _tiny_panel():
    d = pd.to_datetime(["2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05"])
    spot = [100.0, 101.0, 100.0, 99.0]
    mids = [2.00, 2.50, 2.60, 3.00]
    rows = []
    for k in range(4):
        rows.append(dict(date=d[k], symbol="X", expiry=pd.Timestamp("2026-04-30"), strike=100.0, spot=spot[k],
                         bid=mids[k] - 0.05, ask=mids[k] + 0.05, mid=mids[k], spread=0.1, dte=60, tau=60 / 365,
                         m=spot[k] / 100.0, scale=0.05, theo=3.0 if k == 0 else mids[k], delta=-0.5, model_delta=-0.4,
                         q=1.0, bucket=2))
    return pd.DataFrame(rows)


def test_backtest_pnl_arithmetic():
    p = _tiny_panel()
    cfg = BacktestConfig(rule="spread", horizon=2, lag=0, hedge="none", commission=0.65)
    t = run_backtest(p, "theo", cfg)
    assert len(t) == 1 and t.iloc[0]["side"] == 1            # only day 0 has theo above the ask
    assert np.isclose(t.iloc[0]["entry"], 2.05) and np.isclose(t.iloc[0]["exit"], 2.55)   # buy ask, sell bid on day 2
    assert np.isclose(t.iloc[0]["net_pnl"], 100 * (2.55 - 2.05) - 1.30)


def test_backtest_hedge_and_lag():
    p = _tiny_panel()
    cfg = BacktestConfig(rule="spread", horizon=2, lag=0, hedge="vendor", commission=0.0, stock_cost_bp=0.0)
    t = run_backtest(p, "theo", cfg).iloc[0]
    # long a put with delta -0.5: hold +50 shares; spot 100 -> 101 -> 100 so the hedge nets 50 * (+1) + 50 * (-1) = 0
    assert np.isclose(t["hedge_pnl"], 0.0)
    cfg1 = BacktestConfig(rule="spread", horizon=2, lag=1, hedge="none", commission=0.0)
    t1 = run_backtest(p, "theo", cfg1).iloc[0]
    assert t1["entry_date"] == pd.Timestamp("2026-03-03")    # fills one close after the signal
    s = summarise(run_backtest(p, "theo", cfg))
    assert s["n_trades"] == 1 and s["hit_rate"] == 1.0
