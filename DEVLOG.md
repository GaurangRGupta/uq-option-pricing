# Dev Log

Running, chronological session log for the `ivuq` project. Different purpose
from `claude-context.md` (which is a point-in-time snapshot of "what's built
and decided") — this file is the day-by-day record of what happened in each
work session, kept so any of the three of us (or a fresh assistant session)
can pick up context fast without re-reading everything.

Newest entry on top.

---

## 2026-10-25 real-market test: live Yahoo snapshot, report figures, ablation, forward ledger

- **Data sources checked.** (1) yfinance works: real bid/ask for the current chain only, no history, so a history exists only from the day snapshots start (first one: Friday 2026-10-23 close, SPY 1,235 / AAPL 155 / NVDA 161 put quotes, saved under `data/live/snapshots/`, gitignored). (2) London Strategic Edge (`londonstrategicedge.com`, client `lse-data`): free plan, OPRA TRADES since Oct 2025, candles, per-minute Greeks computed twice a day, chain with last trade; NO bid/ask quotes (docs: "No quotes"). So it cannot drive a take-the-quote backtest; planned use is an independent traded-price check and implied-vol history. Needs the user's key in `.env` (`LSE_API_KEY=...`, gitignored); client `src/ivuq/data/sources/lse.py`, probe `scripts/lse_probe.py` (not yet run, no key). (3) ThetaData ($80) is the route to a real multi-year dense-chain NBBO backtest; to be bought by the user; check before buying that the tier includes historical option quotes (bid/ask) for SPY/AAPL/NVDA with at least 2 years of history.
- **Live pipeline.** `src/ivuq/data/sources/yf_snapshot.py` (same schema as `dolthub.clean_puts`; spot = put-call-parity spot because Yahoo's close is stale against after-close option quotes, about $1.1 below on SPY that day), `scripts/live_signals.py` (calibrate on the snapshot with the PINN or FD pricer, BS flat/smile comparison theos, past-only conformal band built from the historical panel plus earlier live dates), `src/ivuq/trading/live.py` + `scripts/live_settle.py` (forward ledger: signals become the same hedged trades and settle against later snapshots; needs one snapshot per trading day). `tests/test_live_data.py`: 6 pass.
- **Live result, one day (2026-10-23, band built from DoltHub history, alpha 0.10).** Coverage of the band: single network 0.745, coupled 0.776, exact solver 0.889 (target 0.90). Mean |theo - mid| on SPY: 1.07 / 1.02 / 0.46 dollars. Reasons: the DoltHub history is a sparse subset (SPY about 85 quotes a day, $15 strike step, 4 expiries) and the live chain is dense (SPY $1 step, 8 expiries) with tighter spreads (AAPL median spread 0.35 against 0.80); the networks are trained on a narrow parameter box (kappa up to 6, xi up to 1.2) and cannot fit the dense chain as well as the exact solver's wide box. Live signals cluster in contiguous strike blocks (model misfit, not isolated mispricings). No forward P&L yet: the first settlement needs the 3rd snapshot after 2026-10-23.
- **Report figures and tables.** `src/ivuq/reporting/{analysis,figures}.py`, `scripts/make_report_figures.py` -> `results/heston_trading/figures/` (35 PNG, `INDEX.md` explains each), `ablation_table.csv`, `cost_grid.csv`, `monthly_pnl.csv`, `gap_closure_slopes.json`. Ablation (theo x band, development/test split at 2026-05-01, same close): with band, test: single PINN +$9.5k (196 trades, Sharpe 6.0), coupled +$8.3k, BS flat +$6.3k, exact solver -$2.9k, BS smile -$0.2k; without band every theo loses in test (-$5k to -$14k). In development the single PINN with band is only +$274 (CI of mean daily P&L [-106, 99]), so the in-sample profit is almost all in May to October 2026. Next close: all negative in both periods.
- **Correction to an earlier number.** Next-day gap closure was reported as slope 0.11 (Heston) and 0.02 (BS flat). Recomputed with an explicit definition (y = next-day mid change net of delta times stock move, over c; x = (theo - mid)/c; consecutive trading days; |x|,|y| < 15): Heston 0.11 (se 0.01), BS flat 0.09 (se 0.01), BS smile 0.50 (se 0.04, its theo hugs the mid so gaps are tiny). Heston and flat have the same, small reversion, which supports the reading that the same-close profit is quote noise, not dynamics. The earlier 0.02 for BS flat is superseded.
- **Bug found and fixed while building the figures:** `add_rolling_conformal` writes the conformal quantile into column `q`, overwriting the dividend yield `q` of the panel; any code using `q` as the yield after calling it gets NaN/inf. The new code keeps the quantile as `cq`. Existing study outputs were not affected (the yield is not used after that call there).
- **Next:** (1) user to put the LSE key in `.env`, then run `scripts/lse_probe.py`; (2) daily `scripts/live_snapshot` step (take_snapshot, live_signals, live_settle) after each close to grow the forward test; (3) ThetaData history for a dense-chain backtest and recalibration of the band on it; (4) widen the network training box and retrain; (5) interactive dashboard (Streamlit + Plotly) once the user confirms.

## 2026-10-23 16:30 wrap-up: dashboard, paper draft, tests, commit blocks

- **Dashboard** (Artifact, type Dashboard, private): https://claude.ai/artifact/8kq82LruZzR38uczKcWAtC . Six attached CSV datasets built by `scratchpad/dash_data/prep.py` from `results/heston_trading` (oos, insample, coverage, accuracy, signals, equity). Sections: out-of-sample table, cumulative P&L, band coverage, latest signals with model tabs, in-sample table, network accuracy. The page's automatic check document had not appeared when I last looked (needs the dashboard open in a browser), so its rendering is unchecked. Data are a fixed copy, not live: rebuild with the prep script and re-upload if results change.
- **Paper draft** in `paper_heston/` (`main.tex`, `references.bib`, `make_tables.py`, generated `tables/*.tex`, `numbers.tex`). No LaTeX compiler on this machine, so it is not compiled. Tables are generated from the result files. Five bibliography entries (`new_refs.bib`) are not verified against Crossref/arXiv. Text states the null result.
- **Full test suite:** `python -m pytest -q`: 142 passed (7 min 28 s). Old BS PINNs, ML/DL baselines and `ivuq.uq` are untouched.
- **Extra findings used in the paper.** Exact-solver calibration pins kappa at its upper bound on 64% to 74% of days and xi at its upper bound on 35% to 71%; the PINN calibration pins xi and rho on 11% to 54% of days. Out-of-sample bootstrap intervals for mean daily P&L at lag 0 exclude zero for the single PINN [168, 670], coupled PINN [178, 665], BS flat [124, 358], BS smile [35, 304] and include zero for the exact solver [-95, 150]; at lag 1 all are negative except the single PINN [-234, 10]. Lag 0 test profit is about $1 per share, roughly the entry edge, mostly on AAPL and NVDA (SPY has few trades). Next-day gap closure regression: slope 0.11 for Heston, 0.02 for BS flat, so the quote reversion is concentrated in the selected extreme quotes.
- **README** gained a Heston section; test count there updated to 142. `.gitignore` now ignores `results/heston_trading/signals_*_h_full.csv` and `trades_*.csv`.
- **Commit blocks 40 to 50** are in `planning/COMMIT_COMMANDS.md` (nothing committed or pushed). `results/figures/fig10_price_slice.png` shows as modified (a script or test regenerated it); not included in any block.
- **Not done:** FD slow mode, coupled-PINN slow mode, SPX (not in data), intraday data. A jump component (Bates) is the obvious next model if the user wants to continue.

## 2026-10-23 15:00 results: backtests, out-of-sample, conformal coverage

Tags: `pinn` = single conditioned network (control variate, anchored), `cpl1000` = coupled solution+boundary network (anchored), `fd` = exact FD solver. Variants: `h_full` (daily 5-parameter fit), `h_slow`, BS `flat`, BS `smile`. Files in `results/heston_trading/`: `backtest_summary_<tag>.csv`, `random_control_<tag>.csv`, `conformal_diagnostics_<tag>.csv`, `oos_summary_<tag>.csv`, `signals_<tag>_h_full*.csv`. FD slow mode was not run.

- **Band coverage is right.** Realised miss rate of the past-only rolling Mondrian band, target 10%: 10.4% to 10.7% for Heston variants, 10.8% to 11.1% for BS flat, by symbol 10.3% to 11.0%, by bucket 10.1% to 11.9%; with the adaptive alpha it is 9.9% to 10.2%. The band is calibrated as a coverage statement about |mid - theo|.
- **Same-close fills (lag 0), horizon 3, in sample, conformal rule:** pinn h_full 631 trades +$15.5/trade (+$9.8k, Sharpe 2.1); cpl1000 h_full 624 trades +$18.2 (+$11.4k, Sharpe 2.7); fd h_full 404 trades -$7.6 (-$3.1k); BS flat +$11.4 (+$7.6k); BS smile +$1.4. Random trades with the same counts: -$50k to -$58k. Spread-only rule (no band) loses (-$6.6 to -$18/trade).
- **Next-close fills (lag 1):** every theo, every rule, loses money (conformal -$28 to -$56 per trade, hit rate 0.30 to 0.40). Random control -$25k to -$31k.
- **Out-of-sample (settings frozen on dates before 2026-05-01, test from 2026-05-01), lag 0:** pinn h_full +$22.8k on 223 trades, Sharpe 7.2; cpl1000 h_full +$24.1k on 243 trades, Sharpe 7.3; fd h_full +$0.9k on 64 trades, Sharpe 0.9; BS flat +$13.9k on 217 trades (Sharpe 6.5); BS smile +$4.3k. Random control about -$20k each. Lag 1: all negative (pinn h_full -$7.5k, cpl1000 -$19.0k, fd -$9.8k, flat -$11.8k, smile -$13.3k).
- **Reading (honest).** The band-and-theo rule beats the spread-only rule and random trades when fills happen at the same close the theo was fitted on, but that is not an executable edge: the profit disappears as soon as entry moves to the next close. The crude BS flat theo earns a similar lag-0 profit, and the exact FD theo (best fit, MAE vs mid 0.25 to 0.42) earns less than the approximate PINNs. So the lag-0 profit tracks quote noise and bid-ask bounce in end-of-day data, not the quality of the Heston model. Heston-without-jumps does not give a tradable edge on this data. The PINN theos are not better than FD or BS in a statistical sense; differences between pinn, cpl1000 and fd at lag 0 should not be over-read. No edge is a valid result and is what the paper should report.
- **Signals (latest date 2026-10-21, alpha 0.10):** of 268 quotes, pinn 11 SELL / 0 BUY, cpl1000 11 SELL / 1 BUY, fd 5 SELL / 2 BUY; rest HOLD.
- Remaining: dashboard, research paper draft, cleanup (full old test suite, README section, COMMIT_COMMANDS blocks 40+).

## 2026-10-23 13:30 progress note (after compaction)

- FD full-mode theos merged: `data/dolthub/slices/theo_{SPY,AAPL,NVDA}_fd_full.csv` (21,147 / 18,587 / 16,216 quotes). MAE vs mid: SPY 0.415, AAPL 0.285, NVDA 0.250 (dollars).
- Coupled (solution + boundary) models trained with FD price and boundary anchors (3000 epochs, w_data 1000, w_bdata 200), error vs fresh FD solves, $ per $100 strike (MAE / p95 / ATM MAE / bias): SPY 0.045/0.171/0.077/+0.006, AAPL 0.065/0.232/0.077/+0.006, NVDA 0.051/0.152/0.051/+0.002. Coupled SPY is slightly better than the single control-variate network (0.051). Models: `models/heston_cond/<SYM>_cpl1000.pt`. Anchored, not pure physics.
- Launched: `run_heston_study.py --tag pinn --seeds 30`, `--tag fd --seeds 30`, `run_oos_test.py --tag fd`, and coupled theos (`run_heston_theo.py ... --tag cpl1000 --mode full`). Stale unsuffixed outputs from the dry run were deleted.
- Next: OOS for pinn and cpl1000, `make_signals.py`, diagnostics review, dashboard, paper draft, cleanup (full old test suite, README, COMMIT_COMMANDS blocks 40+).

## 2026-10-23 — Heston American-put PINN on real quotes: theo, edge, conformal band, backtest

Direction set by the user (supersedes the paper work for now; paper merge and
the IEEE draft are parked). Full plan: `planning/HESTON_TRADING_PLAN.md`
(local, gitignored). This entry is the running history: it is appended to
after every milestone so a compacted session can resume from here.

### Goal in plain terms
Use a Heston (stochastic variance, no jumps) PINN as the pricer on real
market option quotes, the way a desk uses a theo:
theo = model value; edge = market quote minus theo (market above theo means
rich, below means cheap); a conformal band around theo decides whether the
edge is large enough to trust; trade only outside the band; backtest with
"take liquidity at the quote" fills (no market-making fills, data is end of
day). Second novelty claim: Heston PINN for American options validated on
real market data (earlier claim: the physics residual as a conformal score).

### Decisions already made
- Data: DoltHub `post-no-preference/options` (AAPL, NVDA, SPY, American,
  end-of-day bid/ask), joined to `stocks` (close, dividends) and `rates`.
  Not on this machine: the earlier clone lived on another box, and there is
  no `dolt` here. Pull through the DoltHub SQL API one date at a time (single
  date queries are fast; symbol-range queries time out). Window starts
  2025-10-16; last date in the DB is 2026-10-21 (2026-10-22 returns 0 rows).
  Raw pulls go to `data/dolthub/slices/` and are not committed (CC BY-SA 4.0,
  size).
- Calibration is walk-forward by day: for each date, fit Heston to that
  day's quotes; new days extend the history as data arrives.
- Puts only (the Heston American PINN is put-only; call early exercise
  needs dividends and is out of scope).
- Old code must not change behaviour: the BS PINNs, the ten ML/DL baselines
  and `ivuq.uq` stay as they are and serve as benchmarks. All new work goes
  in new modules and new config classes; the existing test suite is the
  regression check (baseline run started 2026-10-23, result below).
- Environment: torch 2.9.0 CPU build on 16 cores. GPU exists (RTX 3050,
  4 GB) but torch cannot see it; do not reinstall torch without asking.

### Known risks (stated up front, not hidden)
- PINN accuracy has to be cents on a $100 strike or the conformal band will
  be wider than any real edge and the strategy will rarely trade.
- A model fit to the day's own quotes agrees with them by construction;
  residual edges are partly calibration noise, bid-ask and the missing jumps.
- End-of-day quotes: take-the-quote backtest only. Fills are an assumption.

### Progress log (append below, newest last)
- 2026-10-23: audited repo. Heston code exists and is fixed-parameter:
  `pinn/heston_pde.py`, `heston_free_boundary.py` (dual network, put only),
  `american_heston.py`, `pricing/heston.py` (semi-analytic European),
  `pricing/heston_lsmc.py`. No finite-difference Heston pricer, no real-data
  loader, no parameter-conditioned PINN, no trading code. Starting Phase 1
  (data loader).
- 2026-10-23: Phase 1 written: `src/ivuq/data/sources/dolthub.py` (per-date API
  pull, resumable; `clean_puts`), `scripts/pull_dolthub.py`; pull running in
  the background into `data/dolthub/slices/` (gitignored). No `dolt` on this
  machine, so the API is the route (about 4 s per weekday).
- 2026-10-23: Phase 2 done: `src/ivuq/pricing/heston_fd.py` (Craig-Sneyd ADI,
  Rannacher start, projection for the American put; one solve gives the whole
  u(m, v, tau) surface, 0.4 s on the default grid), `tests/test_heston_fd.py`
  (5 pass), `scripts/validate_heston_fd.py`, `results/heston_trading/fd_validation.csv`.
  European vs semi-analytic over 100 random parameter points: mean abs error
  0.4 cents per $100 strike, p95 0.95 cents, max 2.2 cents. American vs LSMC:
  FD is above LSMC by 0 to 2.7 cents (LSMC is a lower bound with a few cents
  of noise). xi->0 early-exercise premium matches CRR to under 1 cent.
  Linear interpolation in x costs about 0.3 cent ATM at short tau; delta uses a
  1% spot bump to stay clear of the interpolation cell.
- 2026-10-23: Phase 1 done. Pulled 2025-10-16 to 2026-10-21 (251 trading days) via the API into
  `data/dolthub/slices/raw` (gitignored); `puts_clean.csv`: 55,950 American put quotes
  (AAPL 18.6k, NVDA 16.2k, SPY 21.1k; 40 to 96 per symbol-day; DTE 10 to 66; m 0.80 to 1.25).
  Drop log: 68,229 puts -> 67,889 bid>0 -> 67,805 uncrossed/unlocked -> 57,648 in the moneyness
  window -> 55,950 above intrinsic. r in [3.57%, 4.11%], q: SPY 1.0%, AAPL 0.36%, NVDA 0.08%.
  ATM implied vol: SPY 0.15 (5th-95th 0.11 to 0.21), AAPL 0.25, NVDA 0.40 (0.31 to 0.51). Median
  relative spread: SPY 1.8%, NVDA 2.7%, AAPL 7.1%. `tests/test_dolthub_loader.py` (3 pass).
- 2026-10-23: Phase 3 code written: `pinn/heston_cond.py` (9-input network over m, v, tau, kappa,
  theta, xi, rho, r, q; LCP min-form loss; optional FD anchor `w_data`, off by default),
  `pinn/heston_cond_data.py` (FD labels, held-out evaluation), `scripts/train_heston_cond.py`.
  Pilots (SPY box, 3000 epochs, pure physics vs anchored) were started; results go below.
- 2026-10-23: Phases 4 to 6 code written, not yet run on real quotes: `calibration/heston_calib.py`
  (robust weighted fit, soft-L1, error scale = half-spread floored at 0.3 vol points of vega),
  `trading/theo_pipeline.py` (mode A full daily fit, mode B slow parameters + daily v0),
  `trading/online_conformal.py` (past-only rolling Mondrian quantile + adaptive alpha),
  `trading/backtest.py` (take-the-quote, delta hedge, costs), `trading/bs_theo.py` (BS flat and
  smile benchmarks). `tests/test_trading.py` (5 pass).
- 2026-10-23: also written, not yet run: `trading/controls.py` (random-signal control),
  `scripts/run_heston_theo.py` (daily theo per symbol -> `data/dolthub/slices/theo_<sym>_<tag>_<mode>.csv`),
  `scripts/run_bs_theos.py` (BS flat/smile theos -> `slices/bs_theos.csv`). The benchmark job was
  started in the background; the two SPY pilots (`models/heston_cond/SPY_phys_pilot.pt`,
  `SPY_anch_pilot.pt`; results in `results/heston_trading/cond_eval_SPY_*.json`) are also running.
- Design conventions to keep consistent (code and docstrings match these):
  * Error scale c_i = max(half-spread, 0.003 * BS vega), used in the calibration weights, the
    conformal score s = |mid - theo| / c and the trade threshold.
  * Trade rule: buy if theo - ask > q*c, sell if bid - theo > q*c (conformal); `spread` rule is
    q = 0; `fixed_k` is k*c. Top 10 trades per symbol-day by (edge - thr)/c.
  * Hedge: vendor delta by default (same for every theo compared), model delta as a variant;
    dividends added to the hedge's total return. Entry lag 0 (same close, optimistic) and lag 1
    (next close, realistic) both reported. Horizons 1, 3, 5 trading days.
  * Calibration box: r in [0.034, 0.043]; q boxes SPY (0.009, 0.015), AAPL (0.002, 0.006), NVDA
    (0, 0.003); theta boxes SPY (0.01, 0.09), AAPL (0.02, 0.16), NVDA (0.06, 0.40); tau_max 0.30.
    (The training script still has the wider placeholder r and q boxes: update before the final
    runs; NVDA v_max planned 0.9.)
  * FD is used to validate the PINN and to price the PINN-calibrated parameters as a classical
    check, not to calibrate daily (about 100 s per symbol-day, too slow).
  * Pure-physics versus FD-anchored PINN: report both; the anchored one is never called a pure PINN.
- Pending, in order: (1) read the pilot errors vs FD; (2) fix the boxes and train SPY, AAPL, NVDA;
  (3) run `run_heston_theo.py` for modes full and slow; (4) join BS theos; (5) rolling conformal +
  backtests (rules, horizons, lags, random control, bootstrap CIs); (6) tests, full old test suite
  rerun (the pre-change baseline log was cut off, so rerun it fully), README, commit blocks 40+ in
  `planning/COMMIT_COMMANDS.md`.
- Nothing from this entry is committed or pushed. New files so far: `src/ivuq/data/sources/dolthub.py`,
  `src/ivuq/pricing/heston_fd.py`, `src/ivuq/pinn/heston_cond.py`, `heston_cond_data.py`,
  `src/ivuq/calibration/`, `src/ivuq/trading/`, `scripts/{pull_dolthub,validate_heston_fd,
  train_heston_cond,run_heston_theo,run_bs_theos}.py`, `tests/{test_dolthub_loader,test_heston_fd,
  test_trading}.py`, `results/heston_trading/fd_validation.csv`; `.gitignore` gained
  `data/dolthub/slices/`.
- 2026-10-23 (restart): the user is restarting the PC (touchpad fault), which kills every background
  job. State at that moment: the SPY pilots had not reached their first checkpoint (phys at about
  epoch 1750 of 3000, anchored at about 1250; no `.pt` saved, so both are lost) and the BS-theos job
  had written nothing. Saved on disk and safe: `data/dolthub/slices/raw/` and `puts_clean.csv` (the
  pull is complete), all code and tests listed above, `results/heston_trading/fd_validation.csv`.
  To resume from the repo root with `PYTHONPATH=src` (bash `export PYTHONPATH=src`):
  1. `python -m pytest tests/test_heston_fd.py tests/test_dolthub_loader.py tests/test_trading.py -q`
     (should be 13 pass).
  2. Rerun pilots (about 0.25 s per epoch alone, 4 threads; run them one at a time, or two with 4 threads
     each and nothing else heavy, because contention made them 2x slower):
     `python scripts/train_heston_cond.py --symbol SPY --epochs 3000 --w-data 0 --tag phys_pilot --threads 6`
     `python scripts/train_heston_cond.py --symbol SPY --epochs 3000 --w-data 5 --tag anch_pilot --threads 6 --n-label-params 300`
     Each prints the evaluation vs fresh FD solves at the end and writes
     `results/heston_trading/cond_eval_SPY_<tag>.json`. Checkpoints every 2000 epochs.
  3. `python scripts/run_bs_theos.py --workers 8` (about 55,950 quotes, CRR implied vols; run it alone).
  Windows note: scripts that use ProcessPoolExecutor need the `if __name__ == "__main__"` guard
  (already added). The torch here is the CPU build (2.9.0); the RTX 3050 is unused. Foreground `sleep`
  is blocked in the Bash tool, so wait on jobs with a background `until` loop.
- 2026-10-23 (resumed after restart): 13 new tests pass. `train_heston_cond.py` boxes fixed (r 0.034 to
  0.043; q SPY 0.009 to 0.015, AAPL 0.002 to 0.006, NVDA 0 to 0.003; NVDA v_max 0.9). Wrote
  `scripts/run_heston_study.py` (all variants on a common set of quotes; rules conformal, conformal_aci,
  spread, fixed_k; horizons 1/3/5; lags 0/1; random control; diagnostics). Job logs now go to the session
  scratchpad (`.../scratchpad/pilot_*.log`), not /tmp (the bash /tmp view lagged).
  PILOT 1, SPY, 3000 epochs, pure physics (`w_data=0`), vs 40 fresh FD solves x 200 points in the data
  region: MAE $0.67 per $100 strike, p95 $1.84, max $3.63, bias +$0.67 (biased high), ATM MAE $1.34, wings
  $0.47. NOT adequate: on an SPY strike near 700 that is several dollars, larger than the spread. Same
  upward bias as the GBM N2 v1. Conclusion so far: 3000 epochs of pure physics in nine dimensions is far
  from enough; options are more epochs, bigger batches, L-BFGS finish, and the FD anchor (reported as the
  anchored variant). Anchored pilot result: see next line when it lands.
  PILOT 2, SPY, 3000 epochs, FD-anchored (`w_data=5`, 300 label parameter sets): MAE $0.23, p95 $0.98,
  max $2.65, bias +$0.21, ATM MAE $0.66, wings $0.11. Better, still not enough (target about $0.02 to 0.05
  per $100 strike, since an SPY spread is about 1.8% of price). Both files in
  `results/heston_trading/cond_eval_SPY_{phys,anch}_pilot.json`.
- 2026-10-23: control variate added to `pinn/heston_cond.py` (flag `control_variate`, default False,
  eq. (3) in the module docstring): u = BS_put(m, sigma^2 = v, tau) + (tau / tau_max) * N(...). Terminal
  condition is exact and the payoff kink sits in the known part. Check with the correction zeroed: the BS
  part alone is already MAE $0.09 / p95 $0.34 / ATM $0.14 vs FD (7x better than the trained pure-physics
  pilot). `train_heston_cond.py` got `--control-variate`, `--width`, `--n-interior`. Runs started:
  `cv_phys` (w_data 0) and `cv_anch` (w_data 5), SPY, 3000 epochs, 8 threads each; logs in the session
  scratchpad `cv_*.log`, results `results/heston_trading/cond_eval_SPY_cv_{phys,anch}.json`.
  RESULTS (SPY, 3000 epochs, $ per $100 strike vs 40 fresh FD solves; MAE / p95 / ATM MAE / bias):
    pilot physics (no CV)      0.67 / 1.84 / 1.34 / +0.67
    pilot anchored w=5 (no CV) 0.23 / 0.98 / 0.66 / +0.21
    BS part alone (CV, zero correction, untrained) 0.09 / 0.34 / 0.14 / -0.02
    CV physics                 0.24 / 0.76 / 0.42 / +0.24
    CV anchored w=5            0.16 / 0.52 / 0.27 / +0.16
  Reading: the control variate helps a lot, but trained networks are all biased HIGH and worse than the
  untrained BS part. In the anchored run the physics loss (about 2e-4) outweighs the weighted data loss
  (about 5e-5), so the optimiser follows the physics. Next test: `cv_anch100`, `cv_anch1000` (w_data 100 and
  1000, 800 label parameter sets, 8000 label points per epoch). Added `--n-data` to the training script.
  If still not near $0.02 to 0.05, fall back plan: FD solver prices the final theo at the calibrated
  parameters, the network is only the fast calibrator, and the gap is reported.
- 2026-10-23: user pointed to the Coupled-PINNs paper (arXiv 2605.06688, local copy
  `planning/2605.06688v1.pdf`, text extract in the session scratchpad) and repo
  (github.com/Rohan-217/American_Options_Pricing_using_Coupled_PINNs); user wants the Heston American PINN
  built regardless; the BS benchmarks are comparison columns, not a gate. Read both (ideas only, our code
  stays our own). Facts:
  * Their model: ONE trained net per fixed Heston parameter set (K=10, T=0.25, r=0.1, kappa=5, theta=0.16,
    xi=0.9, rho=0.1); solution net (s,v,tau) [32,64,128,64,32] GELU + boundary net (v,tau) with sigmoid-free
    linear output; trained only in the continuation region s > sf with intrinsic value below sf; weights
    pde 10, term 15, spatial 5, value-match 15, smooth-pasting 5; power-law sampling (tau 1.6, s 3), pools of
    80k, batch 8192, Adam 3500 + 3000 epochs, boundary net updated every 10th step; synthetic points only; repo
    prints no error metric.
  * Their paper's accuracy (Table 3, K=10, vs Ikonen-Toivanen): absolute error 0.0001 to 0.0136 dollars, i.e.
    about $0.001 to $0.14 per $100 strike, with ATM-ish errors 0.006 to 0.013 (about $0.06 to $0.13 per $100).
    So a dedicated single-parameter PINN is itself only good to roughly 5 to 13 cents per $100 strike. Our
    parameter-conditioned net at $0.16 MAE (anchored, 3000 epochs) is in the same range; a realistic target is
    about $0.05, and cent-level accuracy is not something PINNs deliver here.
  * Ideas worth trying for us: (a) continuation-only training with intrinsic value below a learned boundary
    removes the LCP min-form bias (our `heston_free_boundary.py` already does this, but for one parameter set);
    a parameter-conditioned dual-network variant is the next candidate; (b) power-law collocation near the
    boundary and near maturity; (c) large resampled pools with big batches.
- 2026-10-23 (user: finish fast, then trading signals, out-of-sample tests, dashboard, papers):
  * Built the coupled model the user asked for as inspiration: `pinn/heston_cond_coupled.py`
    (parameter-conditioned solution net with the BS control variate + boundary net b(v, tau, p);
    PDE only in the continuation region, value matching + smooth pasting at m = b, optional FD anchors on
    prices and on the FD exercise boundary), `fd_labels_coupled` in `heston_cond_data.py`, `--coupled`
    in `train_heston_cond.py`, `tests/test_heston_cond.py` (5 pass). Run `cpl_anch` (SPY, 3000 epochs,
    w_data 100, w_bdata 50, 600 label sets) is training; results go to
    `results/heston_trading/cond_eval_SPY_cpl_anch.json`.
  * KEY FINDING for the trading use: PINN accuracy (best so far $0.16 per $100 strike) is far coarser than
    SPY spreads (about $0.07 to $0.20 on a $700 strike), and the 12-day SPY smoke run showed PINN theo MAE
    $0.76 vs mid. The PINN price alone cannot resolve mispricings. Decision: the final theo is the
    finite-difference solution at the calibrated parameters (FD mean error 0.1 to 0.5 cent per $100, grid
    "fine"/"mid"); the PINN is the fast amortised calibrator and is reported with its own direct-theo
    variant and measured error. `FDPricer` (calibrate with mid grid, reprice with fine grid, surface cache,
    r and q rounded to 5 bp), `run_theo(final_pricer=, warmup_dates=)`, `pinn/heston_boxes.py`,
    `scripts/run_heston_theo.py` rewritten (`--pricer pinn|fd`, `--start/--end/--warmup` chunks).
  * Smoke results: PINN-calibrated theo on 12 SPY days ran in 26 s. FD-calibrated theo on 6 SPY days took
    6.8 min (contended CPU, about 70 s per day: needs chunked parallel runs once training jobs stop). Even the
    FD fit on day 1 left RMS(model - mid)/c = 5 and parameters on the box edges (xi 1.2, rho -0.95): Heston
    without jumps struggles with SPY's steep short-dated skew. This is a model limit to report, not a bug.
  * User question answered: the FD solver is the classical grid solver for the same Heston American-put
    PDE; it exists as the PINN's answer key, label source, precise theo and benchmark.
- 2026-10-23 research framing, in the user's words (keep): Black-Scholes, the BS PINN and the ML/DL
  baselines are there for COMPARISON only; it does not matter if they price more accurately, because they
  carry assumptions (constant volatility among others) that real markets break, while Heston / Heston PINN
  models volatility as its own mean-reverting random process correlated with the stock (no jumps) and so
  suits live markets better. That is why it is being built. Research plan: Heston PINN + conformal
  uncertainty on a real options dataset, then use it for trading (backtest, out-of-sample testing, trading
  signals), then dashboard and papers. Write-up caution agreed: Heston is "fewer, more realistic
  assumptions", not assumption-free (CIR variance, parameters fixed within a day, no jumps); misfits such as
  the short-dated SPY skew pushing parameters to the box edges are reported as findings.
- 2026-10-23 9:40 RESULT, SPY anchored runs with the control variate (3000 epochs, 800 label sets, 8000 label
  points per epoch), $ per $100 strike vs 40 fresh FD solves (MAE / p95 / ATM MAE / wings / bias):
    w_data 100  : 0.061 / 0.227 / 0.123 / 0.044 / +0.025   (`cond_eval_SPY_cv_anch100.json`)
    w_data 1000 : 0.051 / 0.184 / 0.096 / 0.038 / +0.015   (`cond_eval_SPY_cv_anch1000.json`)  <- chosen recipe
  This matches the accuracy of the Coupled-PINNs paper (5 to 13 cents per $100). It is the ANCHORED variant
  (FD labels weight 1000 next to the PDE loss): report it as a physics-regularised surrogate, never as a pure
  PINN; pure-physics result stays the ablation (0.24 with control variate, 0.67 without). Models:
  `models/heston_cond/SPY_cv_anch1000.pt`. AAPL and NVDA training launched with the same recipe
  (`AAPL_cv_anch1000`, `NVDA_cv_anch1000`, logs `aapl_cv1000.log`, `nvda_cv1000.log` in the scratchpad).
  COUPLED SPY run `cpl_anch` (solution + boundary nets, CV, w_data 100, w_bdata 50, 600 label sets, 3000
  epochs): 0.073 / 0.275 / 0.110 (ATM) / bias -0.011 (`cond_eval_SPY_cpl_anch.json`, model
  `models/heston_cond/SPY_cpl_anch.pt`). Slightly less accurate than the single-network w=1000 model
  (0.051) but nearly unbiased and it outputs an explicit exercise boundary b(v, tau, p). Main pricer stays
  `cv_anch1000`; the coupled model is the secondary variant (and the one aligned with the Coupled-PINNs
  idea) and can be retrained with w_data 1000 if time allows.
  AAPL and NVDA, same recipe (cv + w_data 1000, 800 label sets): AAPL 0.062 / 0.216 / ATM 0.092 / bias +0.014;
  NVDA 0.058 / 0.177 / ATM 0.065 / bias +0.021 (`cond_eval_{AAPL,NVDA}_cv_anch1000.json`,
  `models/heston_cond/{AAPL,NVDA}_cv_anch1000.pt`). All three symbols have a pricer at about 5 to 6 cents per
  $100 strike. Launched at 10:05: PINN theos (`--pricer pinn --tag pinn`, modes full and slow) for AAPL and
  NVDA (SPY full finished: n=21,147, MAE vs mid $0.91; SPY slow was running) and the BS theos
  (`slices/bs_theos.csv`). Next: FD-polished theos in parallel date chunks.
- 2026-10-23 10:30: PINN theos done for all 3 symbols, modes full and slow
  (`slices/theo_<SYM>_pinn_{full,slow}.csv`). FD calibration uses the WIDE box (`box_for(sym, wide=True)`:
  kappa 0.3 to 12, xi 0.1 to 2.5, rho -0.99 to 0.2, theta cap 1.6x) because the PINN's narrow box pinned the fit
  (SPY 2025-12-01..05: fit rmse 3.95 -> 2.74 half-spread units, MAE vs mid $0.34 -> $0.26) and parameters still hit
  the new limits (kappa 12, xi 2.5): short-dated SPY skew needs jump-like behaviour, a Heston-without-jumps limit
  to report. FD run: `scripts/run_fd_chunks.py` (24 chunks = 3 symbols x 8, mode full, 10 workers, log
  `fd_full.log`, merged to `slices/theo_<SYM>_fd_full.csv`); slow mode with `--refit-every 5` after that.
  BS theos done (`slices/bs_theos.csv`; own American IV solved for 92% of quotes): BS flat MAE $0.75 /
  median $0.45; BS smile (quadratic per expiry) MAE $0.255 / median $0.055 vs mid. NOTE the smile fit uses
  about 12 free numbers a day against Heston's 5, so it fits the day's quotes better in-sample; the test is
  out-of-sample/dynamics, not fit. `scripts/run_heston_study.py --tag pinn --seeds 3` started as a dry run
  (log in `tasks/`).
- 2026-10-23 11:30 new code (not yet run on FD theos): `src/ivuq/trading/study.py` (shared `load_panel`: quotes +
  all theo variants on a common set of quotes), `scripts/run_oos_test.py` (freeze rule settings on dates before
  --split 2026-05-01, report the test period with block-bootstrap CI and random control),
  `scripts/make_signals.py` (daily BUY/SELL/HOLD table with band, edge, hedge shares),
  `scripts/run_fd_chunks.py`. `run_backtest` rewritten with dict lookups (the pandas-lookup version took more
  than 30 min for the study grid); `tests/test_trading.py` still passes. FD chunks: 24 jobs, none finished after
  35 min (about 45 s per day under load); expect about 100 min total. The first PINN-theo dry run of the study was
  killed (old slow backtest) and restarted with the new code: `study_pinn.log`.
- 2026-10-23 11:35 PRELIMINARY, PINN theos (tag pinn), h_full, horizon 3, 2 random seeds, hedge = vendor delta
  (`results/heston_trading/backtest_summary.csv`, `random_control.csv`; will be overwritten by the final study):
    conformal lag 0: 631 trades, +$15/trade, total +$9.8k; lag 1: 352 trades, -$42/trade, total -$14.8k
    spread-only rule lag 0: 1636 trades, -$6.6/trade; lag 1: -$29/trade
    random signals (same counts): lag 0 -$77/trade, lag 1 -$65/trade (this is the spread cost)
  Reading (not a conclusion): the band improves on the no-band rule and beats random by about $90 per trade at
  same-close fills, but after a one-day delay the signal does not cover the spread. The real study uses the FD
  theos, 30 seeds and the out-of-sample split. FD chunks 10/24 done at 11:35.
- 2026-10-23 11:50 decision: the coupled solution + boundary model is REQUIRED, not optional (the user asked for the
  coupled-PINN design as inspiration; it gives the explicit exercise boundary the single network cannot). Launched
  `--coupled --w-data 1000 --w-bdata 200 --n-label-params 800`, 3000 epochs, 3 threads each, tag `cpl1000`, for
  SPY, AAPL and NVDA (logs `cpl1000_<SYM>.log`; results `cond_eval_<SYM>_cpl1000.json`; models
  `models/heston_cond/<SYM>_cpl1000.pt`). It runs next to the FD chunks (13/24 done at 11:43), which slows them.
- 2026-10-23 12:00 CORRECTION to the earlier "final theo = FD" line (user objected: then what is the PINN for?).
  The PINN theo is the MAIN model for signals and the backtest (single network, and the coupled network once
  trained), with the conformal band on it. The FD theo runs through the same pipeline as the exact-solver
  control: the PINN-minus-FD gap in trading results is the cost of the PINN's approximation error. BS flat and BS
  smile are comparison rows. Why the PINN is worth having even though it is not more accurate than FD: about
  2 s per calibrated day vs about 45 s; one network for every Heston parameter set; autograd Greeks and gradient
  calibration; the conformal band absorbs its error. Study, OOS test, signals and dashboard lead with PINN rows.

### RESUME HERE (written 2026-10-23 12:15, for a compacted or fresh session)

What the project is now: Heston (no jumps) American-put PINN on real DoltHub quotes (AAPL, NVDA, SPY; 2025-10-16
to 2026-10-21; 55,950 put quotes, 251 days) -> daily walk-forward calibration -> PINN theo -> edge vs market -> past-only
conformal band -> trade only outside the band (take the quote, delta-hedged, costs) -> backtest, out-of-sample test,
signals table, dashboard, papers. Paper merge is parked. BS / BS-PINN / ML baselines are comparison rows only.
MAIN model rows: single conditioned PINN and coupled solution+boundary PINN. FD solver theo = exact-solver control.
BS flat and BS smile = comparison. Never say the final theo is FD or that the PINN is unused (user objected twice).

Where everything is (all uncommitted; nothing pushed; new files only, old modules untouched except additive edits):
  data: `data/dolthub/slices/{raw/, puts_clean.csv, bs_theos.csv, theo_<SYM>_{pinn,fd,cpl1000}_{full,slow}.csv}` (gitignored)
  models: `models/heston_cond/<SYM>_{cv_anch1000,cpl1000}.pt` (+ SPY pilots); results: `results/heston_trading/`
  code: `src/ivuq/{data/sources/dolthub.py, pricing/heston_fd.py, pinn/heston_{cond,cond_coupled,cond_data,boxes}.py,
        calibration/, trading/{theo_pipeline,online_conformal,backtest,controls,bs_theo,study}.py}`
  scripts: `pull_dolthub, validate_heston_fd, train_heston_cond, run_heston_theo, run_fd_chunks, run_bs_theos,
        run_heston_study, run_oos_test, make_signals`; tests: `test_{dolthub_loader,heston_fd,heston_cond,trading}.py`
  plan: `planning/HESTON_TRADING_PLAN.md`; memory: `project_heston_trading.md`.
  Run everything from the repo root with `PYTHONPATH=src`; torch is CPU only; Windows needs `if __name__ == "__main__"`
  in scripts that use process pools; foreground sleep is blocked, use Monitor or background until-loops; the bash
  heredoc sometimes fails on long files, use the Write tool.

Accuracy of the pricers vs fresh FD solves ($ per $100 strike, MAE): pure physics 0.67 -> with control variate 0.24 ->
anchored w=5 0.16 -> anchored w=1000 0.051 (SPY), AAPL 0.062, NVDA 0.058; coupled SPY (w=100) 0.073. FD vs exact Heston
European: 0.1 to 0.5 cent. Coupled-PINNs paper reports 5 to 13 cents, so we are in range. Anchored = physics loss +
FD-label term; always called "anchored", the pure-physics numbers are the ablation.

Running at 12:15 (all background; if lost, rerun, they skip finished work or checkpoint):
  1. FD theos, mode full: `python scripts/run_fd_chunks.py --symbols SPY AAPL NVDA --modes full --chunks 8 --workers 10`
     (24 chunks, 14 done at 12:05, finished chunk files are skipped; merges into `theo_<SYM>_fd_full.csv`). Slow mode
     afterwards only if time: `--modes slow` (uses `--refit-every 5`).
  2. Coupled models: `python scripts/train_heston_cond.py --symbol <SYM> --epochs 3000 --coupled --w-data 1000 --w-bdata 200
     --tag cpl1000 --threads 3 --n-label-params 800 --n-data 8000` for SPY, AAPL, NVDA (about 45 min; evaluation JSON
     `cond_eval_<SYM>_cpl1000.json`). Then theos: `python scripts/run_heston_theo.py --symbol <SYM> --pricer pinn
     --model models/heston_cond/<SYM>_cpl1000.pt --tag cpl1000 --mode full` (and `--mode slow`).
  3. Dry-run study on the PINN theos (`run_heston_study.py --tag pinn --seeds 2`): preliminary only.

Next steps in order: (a) when FD merges exist and the coupled theos are done, run
`python scripts/run_heston_study.py --tag pinn --seeds 30` (then `--tag fd`, `--tag cpl1000`; outputs are now suffixed
with the tag: `backtest_summary_<tag>.csv`, `random_control_<tag>.csv`, `conformal_diagnostics_<tag>.csv`,
`trades_<tag>_...`; the dry run in progress still writes the old unsuffixed names, delete those after);
(b) `python scripts/run_oos_test.py --tag pinn --split 2026-05-01` (and fd, cpl1000); (c) `python scripts/make_signals.py
--tag pinn --variant h_full --all` and the latest-date table; (d) review conformal diagnostics (miss rate vs alpha, per
symbol/month/bucket) and the residual structure of Heston vs BS smile; (e) dashboard (Artifact; load the artifact-design
skill first); (f) research paper draft; (g) final: full old test suite (`python -m pytest -q`; the pre-change baseline
run log was cut off), README section, `planning/COMMIT_COMMANDS.md` blocks 40+ (small commits by sub-component,
Co-Authored-By trailer per the latest reminder), final DEVLOG entry.

Findings so far to carry into the write-up: (1) Heston without jumps fits short-dated SPY skew only with extreme
parameters (kappa 12, xi 2.5 at the wide-box limits; fit RMS 2.7 half-spreads even with the exact solver); (2) BS smile
(about 12 numbers a day) fits quotes better in-sample than Heston (5 numbers): the test is out-of-sample and dynamics;
(3) preliminary PINN-theo backtest (h_full, H=3): conformal beats no-band and random at same-close fills (+$15 vs -$77 per
trade) but is negative at next-close fills (-$42 vs random -$65); real numbers pending; (4) the old backtest was
accidentally slow (fixed with dict lookups).

---

## 2026-09-20 — ML/DL baselines and the PINN comparison harness (American put first)

Motivation (user request): show why a PINN over ordinary ML/DL pricers, with
American options as the main case and implied vol included. The paper draft
(LaTeX, `paper/`, gitignored) and its title were also finished earlier this
stretch: "Physics-Informed Neural Networks for Distribution-Free Uncertainty
Quantification in American Option Pricing". It states plainly that conformal
UQ and real-data validation are not yet done.

### What was built (`src/ivuq/baselines/`, `scripts/`, `tests/test_baselines.py`)

- 10 baselines behind one `fit(X, u)` / `predict(X)` interface, inputs
  (m = S/K, tau), target u = V/K (same variables as the PINN).
  Classical ML: polynomial ridge, kernel ridge (CV-tuned), random forest,
  gradient boosting. Deep learning: tanh MLP (same 4x64 shape as the PINN),
  ReLU MLP, residual SiLU net, Fourier-feature net, partially input-convex
  net (convex in m by construction, Amos et al. 2017), and a BS-embedded
  implied-vol net (learns sigma(m, tau), price = Black-Scholes(sigma)).
  The last two were added after the user objected that a plain MLP is too
  weak a comparator.
- Labels: Black-Scholes closed form (European), 500-step CRR tree (American),
  optional Gaussian label noise.
- Scoring (`metrics.py`), all from one shared finite-difference treatment:
  price error, delta/gamma error, BS PDE residual (European), no-arbitrage
  violation rates (convexity, delta range, below-intrinsic, calendar), and
  for the American put the exercise-boundary error and the implied-vol error
  (price inverted through the 200-step tree, compared with the vol recovered
  from the reference price).
- Regimes (`experiments.py`): labels everywhere, labels only near the money
  and short maturity (extrapolation), noisy labels ($0.50). Label counts
  50/200/1000/3000, 2 seeds. PINNs get no labels, so appear as flat lines.
- Runner `scripts/run_baseline_comparison.py` (American first; American call
  skipped because with q = 0 it equals the European call), figure script
  `scripts/make_comparison_figures.py` (fig1-fig10).
- 16 tests pass. They check the label generators, that the metrics score the
  exact Black-Scholes price as perfect, that a broken price is flagged, that
  the input-convex net is convex even untrained, the exercise-boundary
  extraction, and American IV recovery.

### Results (American put, reporting-config PINN, in dollars on a $100 strike)

The honest reading: the PINN is NOT more accurate than models trained on
labels in this setup.
- Price error: PINN 2.31; kernel ridge 0.005, BS-embedded net 0.009, tanh MLP
  0.23 at 3000 labels. Nine of ten baselines beat the PINN.
- Exercise boundary error is 21 for the PINN vs 0.2 to 2.6 for baselines, and
  IV error about 8 vol points. Part of this is the boundary metric (price
  within $0.20 of intrinsic), which a price sitting ~$2 high never satisfies.
- Where the PINN looks reasonable: it needs zero labels; zero calendar
  violations; outside the labelled region its error (~$2.5) is far below
  kernel ridge ($28) and polynomial ridge ($164), though similar to the
  neural nets and well above the BS-embedded net ($0.11).
- It still breaks convexity on ~29% of test points, like most baselines.

Consequence: do not claim superiority in the paper. Defensible claims are no
labels needed, no calendar arbitrage, no extrapolation collapse of
kernel/polynomial models, and the PDE residual as the signal for conformal UQ.

### Known limits / next steps

- The PINN is likely undertrained (fixed weighting, 6000 epochs); improving
  it and rerunning is the recommended next step.
- American under Heston, the project's centerpiece, is NOT in this comparison
  yet: it needs 3-input baselines, slow LSMC labels and a reporting-scale
  N2-Heston run.
- IV is measured at one fixed vol (0.20). A real IV-surface comparison needs
  a PINN conditioned on vol.
- A first version of the figure script let `$` in axis labels render as math;
  fixed by escaping, figures regenerated.
- `results/` holds the CSVs and PNGs; `results_run.log` and `figs.log` are
  scratch logs and should not be committed.

---

## 2026-09-10 — Phase 3b step 2: N2-Heston, American put under Heston (the centerpiece), plus cleanup

Built the actual centerpiece of Phase 3b: the coupled solution+boundary
network for the American put under Heston, per the architectural call made
at the end of the 2026-08-30 session (dual network, own math/code, not an
extension of the GBM LCP). Also cleared the two long-outstanding loose
scratch files.

- Deleted `_scratch_nb1.ipynb` and `paper4_extracted.txt` outright (both
  untracked, so no git history impact). The notebook was leftover
  exploration code that directly imported a reference repo's own modules
  (`american_utils`, `model_types_am`) and reused its exact class names --
  fine as throwaway scratch work at the time, but the kind of thing that
  should never sit in the repo long-term given the project's own
  cite-ideas-never-copy-code rule, so removed rather than kept around. The
  text file was a stale plaintext dump of one of the 4 reviewed papers,
  already properly indexed (PDF + summary) in `planning/papers/`.

- `src/ivuq/pricing/heston_lsmc.py`: Longstaff-Schwartz Monte Carlo
  (Longstaff & Schwartz, 2001) American-Heston reference pricer -- the
  validation instrument N2-Heston needs, since no closed form exists for
  American options and Heston has no fast lattice method the way GBM has
  CRR. Full-truncation Euler for the variance path (Lord, Koekkoek & Van
  Dijk, 2010, so the CIR process can't go negative between steps), OLS
  regression on a small polynomial basis in (moneyness, variance) at each
  backward step. Classical numerical-finance machinery, unrelated to and
  predating every paper in `planning/papers/PAPER_TRAIL.md` -- same
  "independent reference instrument" role `heston.py` and `binomial.py`
  play elsewhere. `tests/test_heston_lsmc.py`: 6 tests, matches CRR when
  vol-of-vol -> 0, American >= European under Heston (both option types),
  deep-ITM early-exercise premium is positive and nontrivial, q=0 call
  stays close to European. All passing.

- `src/ivuq/pinn/heston_free_boundary.py`: the coupled solution+boundary
  network. Classical value-matching (u(b,v,tau)=h(b)) and smooth-pasting
  (du/dm(b,v,tau)=h'(b)) conditions (McKean 1965; Wilmott/Howison/Dewynne
  ch. 8) at a learned boundary surface b_phi(v,tau) (Heston has variance as
  a second state variable, so the free boundary is a surface, not a curve
  the way it is under GBM) -- `BoundaryNet` wraps a plain `PINN(in_dim=2)`
  with a sigmoid so its output always lies in (0,1), exploiting the
  classical fact that an American put's boundary always sits below the
  strike. Scoped to puts only: an American call with q=0 (this project's
  own default) is never exercised early, which sends that boundary to
  infinity -- unlearnable by a bounded network, and pointless anyway since
  American = European there. Paper 3 in `PAPER_TRAIL.md` makes the
  identical scoping choice for the identical reason, arrived at
  independently here, not copied from it.
  - Found during development, not before: with only the two classical
    conditions plus the terminal/far-field anchors, the trained solution
    network sagged well below intrinsic value in the mid-domain (several
    dollars negative on a $100 strike, well past any reasonable soft-
    training slack) while the PDE fit was still converging -- nothing in
    the dual-network formulation stops that the way the LCP's own obstacle
    term does for GBM/N2, since the continuation-region network has no
    obstacle term at all by construction. Added `obstacle_residual`
    (u >= h(m), restricted to the continuation-region collocation points)
    as an eighth loss term -- not one of the two classical free-boundary
    conditions, but a direct consequence of them (continuation region is
    defined as u > h(m)) and cheap to enforce; weighted at 5.0 (well above
    every other term) once weighting it at parity with the rest turned out
    not to be enough to stop the sag. Documented in both the module
    docstring and `AmericanHestonPINNConfig`'s own comments so the reason
    for that unusually high weight isn't lost.
- `src/ivuq/pinn/config.py`: added `AmericanHestonPINNConfig` (put-only, no
  `option_type` field, unlike every other config in this file).
- `src/ivuq/pinn/train_american_heston.py`: one training loop, eight loss
  terms (PDE + obstacle + value-matching + smooth-pasting + two terminal
  conditions + two far-field conditions), one shared optimizer over both
  networks so their coupling flows through ordinary gradient descent.
- `src/ivuq/pinn/american_heston.py`: `AmericanHestonPINN` wrapper --
  `.price()` is piecewise (intrinsic value below the learned boundary,
  solution network above it), `.boundary()` exposes the learned exercise
  boundary directly in real (S, tau) units, `.pde_residual()` mirrors the
  other wrappers.
- `src/ivuq/pinn/__init__.py`: exports `AmericanHestonPINNConfig`,
  `AmericanHestonPINN`.
- `tests/test_pinn_american_heston.py`: 6 tests, fast config (3 layers/32
  wide solution net, 2 layers/16 wide boundary net, 1500 epochs). Recovers
  LSMC within a loose tolerance; obeys the obstacle condition on average
  (same convention as N2/GBM's own obstacle test, not an exact per-point
  guarantee); boundary stays in (0,1); American >= European Heston; direct
  checks on the value-matching/smooth-pasting residuals themselves and on
  `BoundaryNet`'s sigmoid output range (untrained, math/code-consistency
  only). All passing.
- Full suite: 95 tests passing (83 + 6 LSMC + 6 N2-Heston).
- Not started: Track 2 (conformal/UQ harness), still deferred per user
  instruction. This closes out the base-PINN model ladder (N0/N1 GBM, N2
  GBM, N0/N1 Heston, N2 Heston) -- Section 8's Checkpoint B.

### Pending commits for this session's work

Items 22-28 in `planning/COMMIT_COMMANDS.md`. The two deleted scratch files
need no commit (both were untracked already).

## 2026-08-30 (cont'd) — Phase 3b step 1: N0/N1-Heston, European under Heston

Before attempting N2-Heston (American under Heston, the dual-network
free-boundary problem), went and read the actual source of three of the
four reference repos (forked research, not summarized here) to check we
weren't missing a standard trick or misreading a paper. Findings: our
self-adaptive weighting scheme (`train_american.py`) is faithful in
mechanism to `levimcclenny/SA-PINNs`; the Coupled-PINN Heston repo
(`Rohan-217/...`) uses a genuinely dual solution+boundary network with
classical value-matching/smooth-pasting, not an LCP penalty like ours —
legitimate, different route, but it means our LCP approach never produces
an explicit boundary function, which Phase 5 (exercise-boundary conformal
band) will need; Dhiman & Hu's public American PINN has a real bug (their
inequality-violation loss is computed but never added to the backpropagated
loss, and the dead code that would've added it can't ever fire) — our
`free_boundary.py` is strictly more faithful to the LCP than that published
reference. No copying risk anywhere; all three of our implementations are
structurally independent of the repos they were compared against.

Decision from this: build N2-Heston as a dual solution+boundary network
(own math, own code, independently derived), not an extension of the
single-network LCP approach. Leave N2-GBM exactly as-is — already tested,
and Phase 3 was always scoped as commodity reproduction, not the novel
contribution.

Before touching the dual-network American case at all, built the
validation instrument and the simpler stepping-stone first — same
discipline as Phase 2 (validate against a known closed form before adding
free-boundary complexity):

- `src/ivuq/pricing/heston.py`: Heston (1993) semi-closed-form European
  price via characteristic-function inversion, using the "Little Trap"
  sign convention (Albrecher et al. 2007) for numerical stability. No
  overlap with any of the 4 reviewed papers (none touch Heston's own
  closed form) — this is classical numerical-finance machinery, the same
  role `black_scholes.py` plays for N1 and `binomial.py` plays for N2.
  `tests/test_heston.py`: recovers Black-Scholes within a cent as
  vol-of-vol -> 0, respects no-arbitrage bounds, put-call parity holds
  exactly, price increases with initial variance. All 7 passing.
- `src/ivuq/pinn/heston_pde.py`: non-dimensionalized Heston PDE (same
  m=S/K, u=V/K convention as `black_scholes_pde.py`, v left alone since
  it's already dimensionless), derived from Heston's SDE pair via the usual
  Feynman-Kac argument. Boundary conditions: m=0/m=m_max reused directly
  from the BS case (unchanged by v); v=0 is a natural boundary (the PDE
  degenerates on its own, no separate condition needed or well-posed);
  v=v_max uses the standard artificial d2u/dv2=0 condition from Heston
  finite-difference literature (in't Hout & Foulon).
- `src/ivuq/pinn/network.py`: generalized `PINN` to take an `in_dim`
  parameter (2 for GBM's (m,tau), 3 for Heston's (m,v,tau)) instead of
  hardcoding 2 — one-line change, no behavior change for existing callers.
- `src/ivuq/pinn/heston_collocation.py`, `train_european_heston.py`,
  `european_heston.py`: same shape as the GBM European PINN's
  collocation/train/wrapper trio, extended to the extra v dimension and
  the extra v-boundary loss term.
- `src/ivuq/pinn/config.py`: added `HestonEuropeanPINNConfig`
  (`lambda_pde` is the N0/N1 switch here too).
- `tests/test_pinn_european_heston.py`: 5 tests. One is a pure math/code-
  consistency check with no training involved at all — confirms
  `heston_pde.pde_residual` collapses exactly onto `black_scholes_pde.
  pde_residual` (same network, same inputs) when xi=0 and v is held fixed
  at sigma^2, i.e. equation (4) really does reduce to equation (2) in the
  code, not just on paper. The other four mirror N1's tests: N1-Heston
  recovers the closed-form `heston_price` (loose tolerance, fast config),
  and N1-Heston beats N0-Heston (the PDE-loss ablation). All passing.
- Full suite: 83 tests passing (71 + 7 Heston-pricer + 5 Heston-PINN).
- Not started: N2-Heston (the dual-network American free-boundary PINN
  under Heston — the actual Phase 3b centerpiece), and Track 2, still
  deferred per user instruction.

### Pending commits for this segment (Phase 3b step 1)

Items 14-21 in `planning/COMMIT_COMMANDS.md`. Also revised items 9 and 12
from the Phase 3 batch below, since `config.py` and `__init__.py` each got
touched again by this segment's work before Phase 3's commits were made —
see the notes on those items for why the `git add` lines changed.

## 2026-08-30 — Phase 3 built: N2, the American free-boundary PINN under GBM

- `src/ivuq/pinn/free_boundary.py`: the American option as a linear
  complementarity problem (LCP). Reuses the exact same Black-Scholes
  operator L[u] from `black_scholes_pde.py` (no re-derivation, no
  duplicated autograd machinery) via `pde_residual`. The LCP:
  `L[u] >= 0, u >= h(m), L[u]*(u-h(m)) = 0`, soft-penalized as three
  per-point residuals (`ineq_violation`, `obstacle_violation`,
  `complementarity_violation`) that `train_american.py` squares and
  weights. Terminal condition and both spatial boundary conditions are
  unchanged from the European case (American and European agree at expiry
  and at the domain edges) — reused directly, not redefined.
- `src/ivuq/pinn/config.py`: added `AmericanPINNConfig`, same shape as
  `EuropeanPINNConfig` plus `weighting_scheme: "fixed" | "curriculum" |
  "self_adaptive"` — the Phase 3 ablation from Section 5/6 of the roadmap
  (self-adaptive weighting folded in as a training-config option, not a
  standalone phase).
- `src/ivuq/pinn/collocation.py`: added `sample_interior_near_boundary` —
  rejection-samples interior points where the model's current prediction is
  close to the intrinsic value, i.e. near its current estimate of the
  exercise boundary. Falls back to uniform sampling early in training when
  nothing qualifies yet.
- `src/ivuq/pinn/train_american.py`: one training loop, three schemes:
  - `fixed` — constant LCP loss weights, fresh uniform interior sampling
    every epoch. The baseline.
  - `curriculum` — the three LCP weights ramp linearly from 10% to 100% of
    target over `curriculum_ramp_epochs`; once the ramp completes, interior
    sampling shifts to half uniform / half boundary-concentrated.
  - `self_adaptive` — a simplified, points-only version of McClenny &
    Braga-Neto (arXiv:2009.04544): interior points sampled once and held
    fixed for the run, each of the three LCP terms gets a per-point
    trainable weight (`softplus(a_i)`), updated by gradient ascent (flip
    the gradient sign on those params before the shared optimizer step).
    Deliberately scoped down from the paper's full per-point scheme applied
    everywhere — here it's only the free-boundary terms, since that's the
    one place spatial adaptivity actually matters (concentrating weight
    where the a-priori-unknown exercise boundary sits); terminal/boundary
    weights stay plain scalars, no free boundary to resolve there.
- `src/ivuq/pinn/american.py`: `AmericanPINN` wrapper, same interface as
  `EuropeanPINN` (`.fit()`, `.price()`, `.pde_residual()`).
- `tests/test_pinn_american.py`: fast config (3 layers, 32 wide, 1200
  epochs), 8 tests, all passing. No closed form exists for American options,
  so the reference is the CRR binomial tree (`ivuq.pricing.binomial.
  crr_price`) — comparing against a tree at all is new relative to the 4
  reviewed papers (none do it). Checks (a) all three weighting schemes
  converge to within a loose tolerance of CRR, for both calls and puts, and
  (b) the obstacle condition `u >= intrinsic` roughly holds after training.
  Full suite: 71 tests passing (63 existing + 8 new).
- At the fast/test config, mean abs error vs. CRR across all six
  (scheme, option_type) combinations ranged ~$3.26–$5.32 on a $100-strike
  option — meaningfully better than an untrained/broken model (10s of
  dollars off), but this is the fast config; no scheme was singled out as
  clearly best at this scale, and that's expected — the ablation is meant to
  show all three are viable, not to crown a winner yet. At the real config
  (4 layers, 64 wide, 6000 epochs, matching N1's reporting config,
  `fixed` weighting): mean abs error vs. CRR ≈ $2.31 (put) / $2.66 (call) on
  a $100-strike option — tighter than the fast-config numbers above, as
  expected, and in the same ballpark as N1's own real-config BS-recovery
  error ($0.21 put / $0.74 call), just a bit looser since the LCP is a
  harder target than plain BS. `curriculum`/`self_adaptive` real-config
  numbers not run yet (4 more full runs at this scale) — can be run on
  request if a head-to-head at full scale is wanted.
- Not started: extending N2 to Heston (Phase 3b), and Track 2
  (conformal/UQ harness) — both explicitly deferred this session per user
  instruction.

### Pending commits for this session's work (Phase 3, N2)

1. `feat(pinn): American free-boundary PDE as a linear complementarity problem` — `src/ivuq/pinn/free_boundary.py`
2. `feat(pinn): American PINN training configuration (weighting-scheme ablation)` — `src/ivuq/pinn/config.py`
3. `feat(pinn): boundary-concentrated collocation resampling` — `src/ivuq/pinn/collocation.py`
4. `feat(pinn): American PINN training loop (fixed/curriculum/self-adaptive weighting)` — `src/ivuq/pinn/train_american.py`
5. `feat(pinn): AmericanPINN wrapper and package exports` — `src/ivuq/pinn/american.py`, `src/ivuq/pinn/__init__.py`
6. `test(pinn): N2 vs. CRR binomial tree, all three weighting schemes` — `tests/test_pinn_american.py`

Commands to follow, same convention as `planning/COMMIT_COMMANDS.md` (git add/commit/push origin dev/push myfork dev per item) — to be added there once this DEVLOG entry is confirmed accurate.

## 2026-08-25 — Cleanup pass: deleted superseded planning docs and claude-context.md

- Deleted from `planning/` (all superseded per the 2026-08-24 "roadmap is the
  only authoritative doc" decision below; none were git-tracked since
  `planning/` is gitignored, so nothing lost on GitHub): `PROJECT_PLAN.md`,
  `ARCHITECTURE.md`, `BUILD_PLAN.md`, `ADDON.md`, `PROJECT_FLOW.md`,
  `flow.md`, `README-mine.md` (all last touched Aug 3-11, predate the
  roadmap), plus `2605.06688v1.txt` (redundant text extract of the PDF
  already sitting next to it) and `New Text Document.txt` (stray junk).
  `planning/` now only holds `RESEARCH_GAP_AND_ROADMAP.md`,
  `papers/PAPER_TRAIL.md` + the 4 source PDFs, and `COMMIT_COMMANDS.md`.
- Deleted `claude-context.md` from disk entirely (not just untracked —
  gone). It was dated 2026-08-13 (session 2) and had drifted into actively
  wrong territory: referenced the five planning docs just deleted above as
  live specs, claimed "no `.gitignore`, no git repo yet," and claimed
  PINN work was halted — all stale. `DEVLOG.md` fully supersedes it as the
  "pick up context fast" doc. Its debugging war-stories (BAW 35x bug, Yahoo
  dividend-yield bug, LSE field-mapping findings) aren't duplicated here,
  but the file is still recoverable from git history on both remotes
  (`origin`/`myfork`) if ever needed — it was tracked and pushed before
  commit `6d05d51` untracked it.
- `resume_extracted.txt` (the PII-flagged scratch file, unresolved across
  several prior sessions) turned out to already be gone — resolved, no
  action needed.
- Updated `.gitignore` to drop the now-dead `claude-context.md` line (file
  doesn't exist anymore, nothing to ignore) and `planning/COMMIT_COMMANDS.md`
  to drop commit 0 (already done as `6d05d51`, and its subject file is now
  deleted outright rather than just untracked).
- Still sitting on disk, not yet decided: `paper4_extracted.txt` (root,
  old scratch file) and the build-artifact caches (`__pycache__/`,
  `.pytest_cache/`, `src/ivuq.egg-info/`) — all gitignored already, safe to
  clear anytime, just not done yet.

## 2026-08-24 — Roadmap adopted as single source of truth; papers indexed

- Decided: `planning/RESEARCH_GAP_AND_ROADMAP.md` is now the **only**
  planning doc the team follows for direction. `PROJECT_PLAN.md`,
  `ARCHITECTURE.md`, `BUILD_PLAN.md`, `ADDON.md`, `PROJECT_FLOW.md`,
  `flow.md`, `README-mine.md` are left on disk (some early scoping ideas in
  them — e.g. the curriculum-learning / adaptive-resampling recipe for the
  American PINN — turned out to independently match published work, which is
  how the roadmap file came about) but are no longer authoritative.
- Added `planning/papers/PAPER_TRAIL.md` — quick-access index (arXiv link,
  local PDF path, GitHub repo) for the 4 papers the roadmap's gap analysis is
  based on. Full per-paper writeup stays in `RESEARCH_GAP_AND_ROADMAP.md`
  Section 2.
- Ground rule reaffirmed: the 3 papers with public repos
  (`pinn_option_pricing`, `PI-ConvTF`,
  `American_Options_Pricing_using_Coupled_PINNs`) are for reading and
  benchmarking only. All of our code is our own implementation — cite ideas,
  never copy code.
- Coding-style contract for this project going forward: plain, readable,
  entry-level-quant-appropriate code (no over-engineered abstractions, no
  defensive handling for things that can't happen) — production-quality in
  the sense of being tested and packaged correctly, not in the sense of being
  over-abstracted. Comments are fine on most lines, but every comment has to
  earn its place: meaningful, human-sounding, simple, says something the code
  itself doesn't already say. No filler restating what a line obviously does.
- Timeline decision: compress the roadmap's week-based phases into a fast
  10-15 working day build (see `RESEARCH_GAP_AND_ROADMAP.md` Section 8),
  instead of the original ~28-week pace. Paper-writing timing (concurrent
  with the build vs. after) is explicitly deferred — not decided yet.
- Added Section 8 to `RESEARCH_GAP_AND_ROADMAP.md`: the compressed day-by-day
  execution plan, with an explicit checkpoint for when the
  conformal/UQ teammate can start work in parallel with the base-PINN track
  (Day 4, once N1/European PINN is emitting point predictions + PDE
  residuals — doesn't need to wait for the American/Heston models), plus the
  git commit/push convention to use once coding starts.
- No code written yet — still pre-Phase-1. Coding starts when explicitly told
  to start.

### Phase 1 + Phase 2 built (same day)

- `paper/related_work.md`: Deep Optimal Stopping (Becker/Cheridito/Jentzen,
  arXiv:1804.05394), Neural Optimal Stopping Boundary (Reppen/Soner/
  Tissot-Daguette, arXiv:2205.04595), self-adaptive PINN (McClenny/
  Braga-Neto, arXiv:2009.04544), Merton (1976), Heston (1993), and the
  style-vs-dynamics citation (Gaudenzi/Stucchi/Spangaro, arXiv:1712.08137),
  plus the 4-paper recap (full version stays in
  `RESEARCH_GAP_AND_ROADMAP.md`).
- `src/ivuq/pinn/`: European PINN under GBM (N0/N1). PDE derivation
  (non-dimensionalized by moneyness m=S/K) lives as a docstring in
  `black_scholes_pde.py`, next to the code that implements each term of it.
  `EuropeanPINNConfig.lambda_pde` is the N0/N1 switch (0.0 = N0, no interior
  physics constraint; >0 = N1).
  - Tried and reverted: hard-enforcing the terminal condition via an ansatz
    (`u = payoff(m) + tau*correction(m,tau)`) so there'd be no terminal loss
    term to weight. Backfired — the payoff's kink at m=1 has to be exactly
    cancelled by `correction` for every tau, which a smooth tanh network
    can't do, and it showed up as a visible error band around m=1 at every
    maturity. Went back to the standard soft-constraint terminal loss
    (same as all 4 reference papers), and instead improved convergence with
    kink-concentrated terminal-point sampling + a learning-rate step decay.
    Left a note in `network.py` explaining why, so nobody reintroduces it
    without re-hitting the same wall.
  - Result at the "real" config (4 layers, 64 wide, 6000 epochs): mean
    abs price error over an interior (S, tau) grid ≈ $0.21 (put) / $0.74
    (call) on a $100-strike option. Not state-of-the-art, but this is
    explicitly commodity-reproduction work (Section 7 of the roadmap), not
    a claimed contribution.
  - `tests/test_pinn_european.py`: fast config (3 layers, 24 wide, 1500
    epochs, ~10s/model), 4 tests, all passing. Checks (a) N1 recovers
    closed-form BS within a loose tolerance, and (b) N1's interior error is
    meaningfully below N0's at identical settings — this is the actual
    ablation evidence that the PDE loss is doing something, not just a
    theoretical claim.
  - Full existing test suite (59 tests) still passes; `pyproject.toml` now
    depends on `torch>=2.4` (CPU build currently installed — machine has an
    unused RTX 3050, switching to a CUDA build deferred until it's actually
    needed, likely Phase 3b/Heston).
- **Checkpoint A reached**: N1 exists and exposes both `.price(S, K, tau)`
  and `.pde_residual(S, K, tau)`. Whoever's doing the conformal/UQ track can
  start now against this model, per Section 8 of the roadmap.

### Repo visibility / commit decisions (same day)

- `planning/` was already gitignored before this session (pre-existing,
  intentional — "kept locally, not pushed to GitHub").
- Decided (reversing the earlier call below): `paper/` is now gitignored too,
  same as `planning/`. Not pushed to GitHub at all for now.
- `DEVLOG.md` (this file) is going in `.gitignore` too, same reasoning as
  `planning/` — it narrates AI-assistant session work and references
  gitignored files, so it'd read oddly to an outside repo viewer. It stays
  as a local-only continuity file.
- `tests/` stays tracked and pushed as normal, always — it's part of the
  actual deliverable (already true for the pre-existing 59 tests), not
  process notes. Never gitignore tests.
- User feedback: don't propose 1-2 "major" lumped commits for a coding
  session's code — split even a single feature into several small commits
  by sub-component (math/PDE, network, config, training loop, wrapper),
  same granularity as docs/tests already get. Saved to memory
  (`feedback-commit-granularity`) so this is the default from now on.
- Revised: `DEVLOG.md` is **not** in `.gitignore` after all — instead it's
  just never `git add`ed, so it stays untracked and local without needing a
  gitignore rule. Same practical effect, simpler.
  Also added `paper4_extracted.txt` (old scratch file) and
  `claude-context.md` (frozen prior-session context file) to `.gitignore` —
  `resume_extracted.txt` (contains PII) still sitting on disk, untouched,
  still unaddressed.
- `claude-context.md` was previously tracked/pushed to GitHub — being
  removed from tracking (kept on disk, added to `.gitignore`) since it's the
  same kind of local session-continuity file as `DEVLOG.md`.

### The 7 commits for this session's work (Phase 1 + Phase 2), to make when ready

Revised from an original 9-commit plan: user decided `paper/` is gitignored
(not pushed to GitHub at all, file stays on disk) and merged the
torch-dependency commit into the PDE commit since they're small and land
together anyway.

1. `feat(pinn): non-dimensionalized Black-Scholes PDE and boundary conditions` — `pyproject.toml`, `src/ivuq/pinn/black_scholes_pde.py`
2. `feat(pinn): network architecture` — `src/ivuq/pinn/network.py`
3. `feat(pinn): collocation point sampling` — `src/ivuq/pinn/collocation.py`
4. `feat(pinn): training configuration (N0/N1 switch via lambda_pde)` — `src/ivuq/pinn/config.py`
5. `feat(pinn): Adam training loop` — `src/ivuq/pinn/train.py`
6. `feat(pinn): EuropeanPINN wrapper and package exports` — `src/ivuq/pinn/european.py`, `src/ivuq/pinn/__init__.py`
7. `test(pinn): N0-vs-N1 ablation and closed-form BS recovery check` — `tests/test_pinn_european.py`

Not part of any commit: `DEVLOG.md` (pending the open question above),
`paper/` (now gitignored, per above), anything under `planning/` (already
gitignored), `paper4_extracted.txt` and `resume_extracted.txt` (old scratch
files, unrelated to this work, still sitting on disk — never asked to
delete them, so still there).

Note: session ended 2026-08-24, this is being read back on 2026-08-25 or
later (user is restarting their PC) — if these commits get made today
instead of the 24th, that's fine and actually helps the "genuine multi-day
pacing" goal in `project-commit-pacing` memory, not against it.
- Standing principle for the whole build, refined: the ideal is still
  **math first, then code** — derive the equation, then implement it. That
  stays the default. But it's not a hard requirement every single time;
  occasionally an architecture/loss choice gets tried empirically first and
  gets formalized into an equation afterward, and that's acceptable too.
  What's non-negotiable is that **the two always match** — no code sitting
  in the repo whose corresponding equation isn't written down somewhere
  (docstring/paper), and no equation in the paper that the code doesn't
  actually implement. Every loss term, every architecture choice, every
  nonconformity score has to trace back to an equation we can point to and
  justify (the PDE, the free-boundary conditions, the conformal guarantee
  proof) — not "this is what worked empirically" with no derivation ever
  written down. This matters for two reasons: (1) it's what makes the code
  reviewable by someone who knows the math rather than a black box, and
  (2) the paper's math section and the code have to describe the same
  model, regardless of which was written first.
