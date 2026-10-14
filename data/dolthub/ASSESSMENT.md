# DoltHub `post-no-preference/options`: fitness assessment for the ivuq paper

Written 2026-10-11/12 (IST). Read-only with respect to `paper_ieee/` and `results/`:
nothing there was changed. Every number below comes from a query or script
named next to it. API = `https://www.dolthub.com/api/v1alpha1/post-no-preference/<repo>/master?q=<SQL>`.
Local = the dolt clone at `/workspace/data_dolthub/<repo>` (commit noted below).

## 1. Verdict in one paragraph

Usable, but only for a **narrow real-market add-on**, not as a replacement for the
synthetic study. The data is free, CC BY-SA 4.0, about 8 GB, and it cloned onto
the box in about 2 minutes, so nobody has to download it by hand. It has end-of-day
bid/ask, a vendor IV and Greeks for AAPL, NVDA and SPY from 2019-02-09 to 2026-10-09,
with daily rows since 2024. It does **not** have SPX, NDX or QQQ. It has no
underlying price, volume, open interest, exercise style, dividend or rate columns.
The chain is thin: about 3 to 4 expiries per day, at most 67 days to expiry, and
K/S mostly between 0.73 and 1.26. Spot, dividends and Treasury rates are in two
sister repos by the same maintainer, also CC BY-SA 4.0, and they join cleanly.
The current PINNs have sigma fixed at 0.20 and r at 0.03, so they cannot price a
real chain directly. Recommended: keep the paper's main claims synthetic. Add, if
the go criteria in Section 7 pass by 1 Nov 2026, one small "real-market coverage
check" on American puts (AAPL, NVDA, SPY) with a chronological calibration/test
split. Do not hold back the ICRCICN submission for it.

## 2. What the repository contains

| Fact | Value | Source |
|---|---|---|
| Tables | `option_chain`, `volatility_history` | docs page https://www.dolthub.com/repositories/post-no-preference/options/doc/master |
| Description (maintainer's words) | "bids, asks, vols, and greeks for US equity options"; "Data doesn't cover all options, but does cover many of them. More recent data is recorded each weekday. Some older data was just recorded Monday, Wednesday, and Friday. The oldest data was just recorded once weekly. The symbols covered ... SPDR market, sector, and industry ETFs as well as the SPY, MDY, and SPSM components." | same docs page |
| Licence | Creative Commons Attribution-ShareAlike 4.0 International (`LICENSE.md`) | same docs page; local `dolt_docs` in options, stocks and rates clones |
| Provenance | **Not stated.** The only other doc is a generic `AGENT.md` (Dolt how-to). Maintainer = `post-no-preference` (post.no.preference@protonmail.com). DoltHub's 2020 blog says the repo was first published 27 Aug 2020 and "Data starts February 2019". Whether `vol`/greeks come from a vendor or the maintainer's own model is not documented, so treat them as vendor-computed and unverified. | API `select doc_name, doc_text from dolt_docs`; API `select * from dolt_log limit 5`; https://www.dolthub.com/blog/2020-09-30-september-dataset-spotlight/ |
| Update cadence | One commit per trading day for each table, about 06:35 UTC (12:05 IST) the next morning, e.g. "option_chain 2026-10-09 update" at 2026-10-10 06:34:51 UTC. 2,138 commits on master. | API `select * from dolt_log limit 5` |
| Commit assessed | `jbslp1mbpecs27bf232dqii0colr9o60` ("volatility_history 2026-10-09 update") | local `dolt log -n 1` |
| Total rows, option_chain | 119,722,778 | local `select count(*) from option_chain` (`q_total_rows.txt`) |
| Total rows, volatility_history | 1,961,327; 1,299 dates; 2,341 symbols | local query on volatility_history |
| Date range | 2019-02-09 to 2026-10-09 | API `select date from option_chain order by date asc/desc limit 1` (`min/max` over the full table timed out on the API) |
| Dates per year (volatility_history) | 2019: 48, 2020: 155, 2021: 156, 2022: 152, 2023: 148, 2024: 183, 2025: 259, 2026: 198 | local `select year(date), count(distinct date) from volatility_history group by 1` |

### option_chain schema (API `show create table option_chain`)

`date` DATE, `act_symbol` VARCHAR, `expiration` DATE, `strike` DECIMAL(7,2),
`call_put` ('Call'/'Put'), `bid` DECIMAL(7,2), `ask` DECIMAL(7,2), `vol` DECIMAL(5,4),
`delta`, `gamma`, `theta`, `vega`, `rho` (DECIMAL(5,4)).
Primary key (`date`, `act_symbol`, `expiration`, `strike`, `call_put`).

| Wanted column | Present? |
|---|---|
| bid / ask | yes |
| implied vol | yes (`vol`, source undocumented) |
| greeks | yes (delta, gamma, theta, vega, rho) |
| volume / open interest | **no** |
| underlying price | **no** (join `post-no-preference/stocks.ohlcv`) |
| expiration, strike, call/put | yes |
| exercise style | **no** (all listed equity/ETF options are American; must be set by us) |
| dividends / rates | **no** (join `stocks.dividend`, `rates.us_treasury`) |
| last trade price, contract symbol, quote timestamp | **no** |

Note: DECIMAL(7,2) caps strike and bid/ask at 99,999.99, so a future SPX-level
symbol would fit, but none is present.

`volatility_history` has per symbol and date: hv_current, hv_week_ago, hv_month_ago,
hv_year_high/low (+dates), iv_current, iv_week_ago, iv_month_ago, iv_year_high/low (+dates).

### Symbols relevant to us

| Symbol | In data? | Evidence |
|---|---|---|
| AAPL | yes, 2019-02-09 to 2026-10-09 | API: 192 rows on 2026-10-09; local volatility_history |
| NVDA | yes, same span | API: 184 rows on 2026-10-09 |
| SPY | yes, same span | API: 204 rows on 2026-10-09 |
| SPX, NDX | **no** | API `... where date='2026-10-09' and act_symbol in ('SPX','NDX',...)` returned none; local volatility_history has no SPX, SPXW, NDX, XSP, VIX |
| QQQ, IWM | **no** | same queries |
| DIA, MDY, XLK | yes | local volatility_history |

So the European leg of the project (SPX/NDX, cash-settled) **cannot** use this
data. SPY is American and is not interchangeable with SPX (the repo's own
`src/ivuq/data/schema.py` says this, and `option_style_for("SPY")` currently raises).

### Sister repos (same maintainer, CC BY-SA 4.0)

| Repo | Tables | Used for | Check |
|---|---|---|---|
| `post-no-preference/stocks` | `ohlcv` (date, act_symbol, open, high, low, close, volume), `dividend` (act_symbol, ex_date, amount), `split`, `symbol` | spot (close), dividends | SPY ohlcv 2011-01-03 to 2026-10-09. Prices are **not split-adjusted** (NVDA close 1208.88 on 2024-06-07, 121.79 on 2024-06-10), which matches the unadjusted option strikes (NVDA strikes 845 to 1570 on 2024-06-07, 85 to 160 on 2024-06-10). Splits table lists AAPL 4:1 2020-08-31, NVDA 4:1 2021-07-20, NVDA 10:1 2024-06-10. |
| `post-no-preference/rates` | `us_treasury` (1m to 30y yields, percent) | risk-free rate | 9,187 rows, 1990-01-02 to 2026-10-09 |

## 3. Can we get it onto the box? Yes, done

Box (checked 2026-10-11 ~23:35 IST): 126 GB disk, 118 GB free before the clone;
15 GiB RAM with only about 3.6 to 3.9 GiB available (other processes hold the rest);
8 cores; network about 50 to 60 MB/s from DoltHub's remote (dolt progress lines).

- Installed dolt 2.4.2 from the official installer
  (`https://github.com/dolthub/dolt/releases/latest/download/install.sh`) into `/usr/local/bin`.
- `dolt clone post-no-preference/options` into `/workspace/data_dolthub/options`:
  **117 s**, exit 0, about 6.0 M chunks. Peak about 14 GB during the clone, **8.2 GB** on disk after.
  `dolt status`: clean, up to date with origin/master.
- `dolt clone post-no-preference/rates`: 12 s, 13 MB. `post-no-preference/stocks`: 4.5 GB on disk, finished in a few minutes (my timer for this one was unreliable).
- Total now under `/workspace/data_dolthub/`: about 12.7 GB. Disk after: 105 GB free.
- Daily refresh later: `cd /workspace/data_dolthub/options && dolt pull`.
- The SQL API alone is **not** a good bulk path: whole-table aggregates time out
  ("context deadline exceeded"), and even `ohlcv where act_symbol='AAPL' and date between ...`
  timed out 4 times out of 4. Queries on a single date are fast. The clone is the right route.

Slices extracted for the project (all under `/workspace/ivuq/data/dolthub/`):

| File | Rows | How |
|---|---|---|
| `slices/option_chain_AAPL_NVDA_SPY.csv` (49 MB) | 550,544 | local `select * from option_chain where act_symbol in ('AAPL','NVDA','SPY')`, 436 s (full scan: the key starts with date, not symbol) |
| `aux/ohlcv_AAPL_NVDA_SPY.csv` | 5,855 | local stocks clone, from 2019-01-01 |
| `aux/dividend_AAPL_NVDA_SPY.csv` | 284 | local stocks clone |
| `aux/split_AAPL_NVDA_SPY.csv` | 4 | local stocks clone |
| `aux/us_treasury.csv` | 1,931 | local rates clone, from 2019-01-01 |

These are large CSVs. Do not commit them to GitHub unless the repo licence notes
are added (ShareAlike, Section 8). Re-create them from the clone with the queries above.

## 4. Data quality (AAPL, NVDA, SPY; scripts `qc_slice.py`, `qc_extra.py`; outputs in `qc/`)

Coverage (`qc/coverage.csv`):

| | AAPL | NVDA | SPY |
|---|---|---|---|
| rows | 181,908 | 185,022 | 183,614 |
| recorded dates | 1,300 | 1,295 | 1,288 |
| trading days in span (from ohlcv) | 1,925 | 1,924 | 1,925 |
| weekend-stamped dates (2019 weekly data, Saturday) | 48 | 48 | 48 |
| median rows/day (all years / 2026) | 140 / 192 | 158 / 170 | 142 / 210 |
| median expiries/day | 3 | 3 | 3 |
| median distinct strikes/day | 30 | 31 | 34 |
| max / median days to expiry | 67 / 28 | 67 / 28 | 66 / 28 |

- 2026: 513 symbol-days have 4 expiries, 76 have 3; DTE quartiles 10/15/31/45/66 days (`qc/extra.json`).
- K/S (using same-day close): 5th to 95th percentile 0.73 to 1.26 (AAPL), 0.73 to 1.27 (NVDA), 0.73 to 1.22 (SPY). The paper's synthetic domain m in [0.6, 1.4], tau up to 1 year is much wider, especially in tau.
- Gaps: weekly in 2019, then Mon/Wed/Fri (about 150 dates/year in 2020 to 2023), daily from 2025. One hole of 14 days, 2024-07-29 to 2024-08-12, in all three symbols (`qc/gaps_over_5d_*.csv`).
- **Market-holiday rows are copies.** Rows exist on exchange holidays (e.g. 2025-12-25, 2026-09-07, 2024-11-28, and 2025-01-09, a market closure). On those dates bid and ask equal the previous record for 100% of matched contracts. These dates must be dropped by keeping only dates present in `stocks.ohlcv`. That removes all 55 (AAPL/NVDA) or 57 (SPY) weekday dates with no same-day close.

Quote quality (`qc/quote_quality.csv`):

| | AAPL | NVDA | SPY |
|---|---|---|---|
| null bid / ask / vol | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| bid = 0 | 1.86% | 0.33% | 3.33% |
| ask = 0 | 0.014% | 0.022% | 0.038% |
| crossed (bid > ask) | 0.003% | 0.0005% | 0.015% |
| locked (bid = ask > 0) | 0.12% | 0.011% | 0.69% |
| median relative spread | 5.4% | 3.7% | 1.7% |
| vol <= 0 (rows) | 22 | 8 | 72 (2 are exactly -1, a sentinel) |
| max vol | 1.91 | 1.66 | 4.30 |
| quote unchanged vs previous record (bid > 0) | 4.3% | 3.7% | 4.1% |

The "unchanged" share includes the holiday copies above. After dropping holidays it
will be lower. It is not proof of staleness on its own.

Spot, dividends, rates:
- Same-day close found for every trading day. 2019 Saturday dates match the prior Friday close (as-of join, 3-day tolerance): 0 missing (`qc/summary.json`).
- Treasury rates missing on 116 of 1,303 option dates. These are weekends, holidays and bond-market-only closures; an as-of (previous day) join fixes them.
- Dividends: cash amounts with ex-dates only. A continuous yield must be built (trailing 12-month sum / spot was used for the checks below) or dividends handled discretely.

When are the quotes taken? On 2026-08-13 the put-call parity spot implied by
DoltHub's nearest-expiry ATM mids is 305.09 (AAPL) and 225.40 (NVDA). The ohlcv
close is 305.26 / 225.30 on 13 Aug and 302.25 / 224.09 on 12 Aug. So DoltHub's
`date` is consistent with **end-of-day quotes of that same date** (`qc/extra.json`).
This is an inference from parity, not a documented fact.

Comparison with the repo's 13 Aug 2026 Yahoo snapshot (`data/raw/*_2026-08-13.csv`):
- Yahoo `underlying_price` equals the **12 Aug** close (302.25 AAPL, 224.09 NVDA), not 13 Aug.
- Yahoo bid > 0 on only 0.1% of AAPL rows, 0.1% NVDA, 0.3% NDX (SPX 99.6%). So the snapshot looks like it was taken outside US trading hours; only SPX has live quotes.
- Same contracts: AAPL 51 matched, NVDA 91 matched (DoltHub has 188 / 184 rows on 4 expiries; Yahoo has 712 / 838 rows on 14 expiries up to 2027-02-19). No matched contract has a positive bid in both, so mids and IVs could not be compared.
- For AAPL and NVDA the DoltHub chain is cleaner than our Yahoo snapshot. It is thinner and shorter-dated.

Arbitrage checker (`src/ivuq/arbitrage/checker.py`, run on DoltHub mids mapped to the
ivuq schema, bid > 0, spot = same-day close, r = 3-month UST, q = trailing dividend
yield; `qc/extra.json`), dates 2024-03-13, 2025-06-11, 2026-08-13, 2026-10-09:
- Monotonicity, convexity, calendar: **0 violations** on all 12 symbol-dates.
- "Parity" flags: AAPL 0 to 5, NVDA 0 to 2, SPY 18 to 32 per date. The checker tests
  the European parity equality. For American options parity only holds as a pair of
  inequalities, and SPY has dividends, so these flags are expected and are not
  arbitrage. An American-parity bound check would be needed before reporting them.
- So yes, the checker runs once a loader supplies `underlying_price`,
  `risk_free_rate`, `dividend_yield`, `option_style` and `market_option_price` = mid.

### Column mapping: DoltHub to ivuq schema (`src/ivuq/data/schema.py`)

| ivuq column | DoltHub source |
|---|---|
| underlying | option_chain.act_symbol |
| option_style | set "american" (must extend `option_style_for` to accept SPY as an American ETF) |
| option_type | lower(call_put) |
| strike | strike |
| expiry_date | expiration |
| quote_date | date (drop non-trading dates) |
| underlying_price | stocks.ohlcv.close on quote_date (unadjusted) |
| market_option_price | (bid + ask) / 2 |
| risk_free_rate | rates.us_treasury, interpolate to expiry, percent/100, convert to continuous |
| dividend_yield | built from stocks.dividend (or discrete dividends) |
| data_source | "dolthub" |
| bid, ask | bid, ask |
| bid_ask_spread | ask - bid |
| vendor_implied_volatility | vol (only for sanity checks, per the schema rule) |
| vendor_delta, vendor_gamma | delta, gamma |
| contract_symbol, last_price, volume, open_interest | not available |

## 5. Fitness for the paper

What it can support: a real-market check of **split-conformal coverage on American
puts** with a **chronological** calibration/test split. Usable volume, trading days
only, puts with bid > 0, from 2025-01-01: AAPL 33,990 quotes on 439 days, NVDA
30,991 on 436, SPY 36,119 on 435 (249 days in 2025, 190 in 2026).

What it cannot support without new modelling:
- The trained PINNs (`src/ivuq/pinn/config.py`) use a scalar `sigma = 0.2` and fixed
  r, q. A real chain has a different vol per day and per strike. So the current N2
  cannot be the base predictor on real data. Options: (a) a sigma-input PINN
  (m, tau, sigma), which is new code, new training and new tests; or (b) a classical
  base pricer (our CRR at our own solved ATM IV per symbol-day). With (b) the
  real-data section tests the conformal layer, not the PINN, and must say so.
- Nothing European (no SPX/NDX). The European N0/N1 results stay synthetic.
- Long maturities: nothing past about 67 days.

Suggested protocol (to be pre-registered in a PROTOCOL file before any coverage is computed):
1. Universe: AAPL, NVDA, SPY American puts; trading days 2025-01-02 to 2026-10-09; drop holiday copies, bid = 0, crossed/locked quotes, vol <= 0, DTE < 7 days; 0.8 <= K/S <= 1.2.
2. Label: mid. Inputs: S = same-day close, r = UST interpolated to expiry, q from trailing dividends.
3. Base predictor: CRR (500 steps) at the symbol-day ATM IV from our own solver (`ivuq.pricing.iv_solver`); optional second predictor, the sigma-input PINN if built.
4. Score: |mid - prediction|; variants: vanilla, normalised by half-spread, Mondrian on {K/S < 0.95, 0.95 to 1.05, > 1.05} x {DTE < 30, >= 30}.
5. Split A (fixed): calibrate on 2025, test on 2026. Split B (rolling): calibrate on the previous 20 trading days, test on the next day, then step forward. alpha in {0.10, 0.05}.
6. Report: marginal coverage, per-bucket coverage, mean width in dollars and in half-spreads, coverage per month (to show drift). State plainly that exchangeability is violated across time, so coverage is not guaranteed.

Risk to the 30 Nov 2026 ICRCICN deadline: moderate with base predictor (b), high with (a).
The current paper is 5 of 6 pages, so there is room for one table and one paragraph,
but every new paragraph also has to go through the team's rewrite for the similarity limit.

## 6. Attribution the paper must give

All three repos are CC BY-SA 4.0. Minimum, in the data section or a footnote:

> Option quotes: "post-no-preference/options", DoltHub,
> https://www.dolthub.com/repositories/post-no-preference/options, commit
> jbslp1mbpecs27bf232dqii0colr9o60 (accessed 11 Oct 2026). Spot prices and dividends:
> "post-no-preference/stocks"; Treasury yields: "post-no-preference/rates" (DoltHub).
> Licensed CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Data were
> filtered and transformed as described in Section X.

- Section 3(a) of the licence: credit the creator, link the licence, say the data were modified.
- ShareAlike (Section 3(b)) applies to shared Adapted Material. If the team publishes the cleaned CSVs (e.g. in the GitHub repo), license those files CC BY-SA 4.0. Keep raw data out of the IEEE PDF (the IEEE copyright form covers the paper) and report only summary statistics there. This is a reading of the licence text, not legal advice.
- No paper or DOI exists for the dataset, so cite it as a web dataset with the commit hash and access date. Do not claim where the maintainer gets the data from.

## 7. Plan of action

| Step | Work | Effort | Go/no-go |
|---|---|---|---|
| 0 | Done: clones, slices, QC (this file) | done | - |
| 1 | Decide with mentor: real-market add-on yes/no; base predictor (b) CRR+own IV, or (a) sigma-input PINN | 1 meeting | If the team wants the PINN itself on real data and cannot spare about a week, choose (b) or skip |
| 2 | Write `src/ivuq/data/sources/dolthub.py` loader (from the local clone or the slice CSVs) to the ivuq schema; add SPY as American in `schema.py`; unit tests on a tiny fixture | 0.5 to 1 day | Loader passes `validate_option_chain` with no missing required columns |
| 3 | Filters + own IV solve per symbol-day; American-parity bound check | 1 day | Own IV converges on >= 95% of filtered quotes; monotonicity/convexity violations near 0 (seen 0 so far); >= 50 usable puts per symbol-day on >= 300 days |
| 4 | Pre-register protocol (Section 5) in `results/uq/paper/real/PROTOCOL.md` before computing coverage | 2 hours | Written and dated before step 5 |
| 5 | Run split A and split B; CSVs + one figure (coverage by month) under `results/uq/paper/real/` | 0.5 day CPU + 0.5 day analysis | Report whatever comes out; no tuning on test months |
| 6 | Add one table + one paragraph + attribution to the paper; regenerate tables from CSV; team rewrites for similarity | 1 to 2 days | Hard stop **1 Nov 2026**: if steps 2 to 5 are not done, keep the paper synthetic-only and list DoltHub in future work |
| (a) only | sigma-input American PINN: config + collocation over sigma, retrain 10 seeds, rerun P2/P4 | 4 to 7 days incl. CPU | Only if started by 20 Oct and the synthetic MAE does not worsen badly |

Independent of all this: a CMT submission for the 15 marks should not wait for the real-data section.

## 8. Files written by this assessment

- `/workspace/ivuq/data/dolthub/ASSESSMENT.md` (this file)
- `/workspace/ivuq/data/dolthub/qc_slice.py`, `qc_extra.py` (read-only checks)
- `/workspace/ivuq/data/dolthub/qc/` (coverage.csv, quote_quality.csv, dates_per_year_*.csv, gaps_over_5d_*.csv, summary.json, extra.json)
- `/workspace/ivuq/data/dolthub/slices/option_chain_AAPL_NVDA_SPY.csv`, `aux/*.csv`, `q_total_rows.txt`
- Outside the project: `/workspace/data_dolthub/{options,stocks,rates}` (dolt clones, about 12.7 GB)
