# Claude Context — read this first after any /compact or new session

This file exists so a fresh Claude session (or a teammate) can pick up this
project with zero prior context. It documents **what has actually happened**,
not what the plan says should happen — check `PROJECT_PLAN.md` for the spec,
this file for ground truth on what's built, tested, and decided.

Last updated: 2026-08-13 (session 2).

---

## 1. What this project is

`ivuq` — a modular, uncertainty-quantified ML framework for options pricing and
implied-volatility surface reconstruction. Three-person capstone. Team:

- **Person A** (you, the user) — math/CS/finance lead. Owns pricing, arbitrage, paper.
- **Person B** — CS + ML math. Wants conformal prediction. Owns the uncertainty layer, neural nets.
- **Person C** — tech only, light finance, not comfortable with heavy math. Owns data/infra/viz.

Priorities, in order: **resume value for quant roles > publish a paper if
possible > eventual reusability.**

Full reasoning lives in `README.md`. The 29-phase spec is `PROJECT_PLAN.md`.
Plain-language walkthrough is `PROJECT_FLOW.md`. Diagrams are `ARCHITECTURE.md`.
Deferred/optional scope is `ADDON.md`. A plain build guide is `BUILD_PLAN.md`.

**Do not assume those five documents are internally consistent with each other
or with this file right now** — see §5 "Known doc inconsistencies" below.

---

## 2. The four claimed contributions (the actual point of the project)

1. The American exercise boundary, with a calibrated uncertainty band (no closed
   form exists for the boundary itself).
2. The coverage-vs-arbitrage-violation frontier — trading off "the band is
   honest" against "the band never implies free money."
3. Physics-informed nonconformity scores: `|y - ŷ| / (1 + κ·|PDE residual|)`,
   using a PINN's own PDE residual as a free difficulty estimator. No
   unstructured model can produce this signal.
4. The leakage demonstration: same data, date-split vs random-split, two very
   different answers — a concrete argument for why Rule 1 (§4) exists.

Everything else in the build serves these four things. When scope pressure
hits, these are what survive; everything else is negotiable (see `ADDON.md`).

---

## 3. Scope decisions (cumulative across sessions)

### 3.1 Dropped India entirely — US-only now

The original plan crossed market (India/US) × style (European/American) into
four groups. **This was based on a factual error.** NSE (India) has issued
*only* European-style options — index AND stock alike — since a SEBI circular
dated 27 Nov 2010, effective 27 Jan 2011. There is no exchange-listed
American-style option in India. The user caught this by challenging a claim
I'd stated confidently; verified via web search, it checked out.

Consequence: the 2×2 factorial never actually crossed (India had no American
cell). Decision: **go US-only.**

### 3.2 SPX vs SPY is not a free substitution — a second, related error

The original docs listed the European group as "SPX (or SPY)" as if
interchangeable. They are not:
- **SPX** (S&P 500 index options) — cash-settled, **European**.
- **SPY** (the ETF tracking the same index) — physically settled, **American**,
  despite tracking an index.
Same split applies to NDX (European) vs QQQ (American), RUT (European) vs IWM
(American). **Always use the index option, never the ETF, for the European group.**

### 3.3 Final locked scope

| Group | Style | Underlyings | Why two per style |
|---|---|---|---|
| European | index, cash-settled | **SPX** + **NDX** | guards against one index's quirks (e.g. NDX's tech concentration) dominating results |
| American | single-stock, physical settlement | **NVDA** + **AAPL** | same guard; also NVDA/AAPL contrast negligible vs real dividend for early-exercise testing |

Encoded in `src/ivuq/data/schema.py` (`EUROPEAN_UNDERLYINGS`,
`AMERICAN_UNDERLYINGS`, `option_style_for()` — raises `ValueError` on anything
else, including SPY/QQQ, on purpose).

### 3.4 Data sources — current status (revised in session 2)

1. **Yahoo Finance** (`yfinance`, unofficial) — primary, and now live-tested
   (see §4.6). Can rate-limit or change shape without notice.
2. **Tradier** — ~~free sandbox fallback~~ **the user corrected this in session
   2: Tradier is actually paid, not free.** Code (`tradier.py`) still exists
   and still has passing mocked tests, but it is **dormant** — not an active
   fallback, no token has ever been obtained, no live test has been or will be
   run against it unless this changes. Do not re-propose it as "the free
   fallback" without re-checking pricing.
3. **London Strategic Edge** (`londonstrategicedge.com`, `pip install lse-data`)
   — free platform, confirmed genuinely distinct from LSEG. **Live-tested in
   session 2 with a real API key the user provided in chat** (see §4.7 for
   what was found/fixed). Key facts now confirmed, not guessed:
   - Real field names: `ticker`, `underlying`, `strike`, `expiry`,
     `contract_type`, `last_price`, `volume_today`, `premium_today`,
     `underlying_price`, `dte`, `iv`, `delta`, `gamma`, `theta`, `vega`, `rho`,
     `last_trade_at`, `updated_at`. No `bid`/`ask`/`open_interest` field exists
     at all (confirmed, not null-but-present).
   - **LSE has zero coverage of SPX or NDX** — checked all 3186 underlyings it
     offers; none are index symbols (SPY/QQQ/TSLA/NVDA/AAPL/etc. only, all
     single-name or ETF). **LSE can only ever be a secondary source for the
     American group (NVDA/AAPL). It is not usable for the European group at
     all.** This is a real scope fact, not a data-quality gap to work around.
   - The API key the user provided (`lse_live_...`) was used only as an
     in-process environment variable for that one live test. **It was never
     written to any file in this repo and is not persisted anywhere.** A
     future session has no access to it — if live LSE testing is needed again,
     the user must provide/set it again.

**Net effect: there is currently no active fallback for Yahoo Finance at all**
(Tradier is paid and parked, LSE doesn't cover the European group and lacks
bid/ask/OI even for the American group). The user explicitly decided to accept
this gap for now rather than adopt CBOE/Alpha Vantage/Polygon (rate limits too
restrictive to be worth it). Revisit only if Yahoo actually breaks.

### 3.5 Column contract — what's required vs optional

Decided explicitly: **if a source is missing bid/ask/open interest/vendor IV,
that's fine, we proceed anyway.** Only actually required per row: identity
(underlying, style, type, strike, expiry, quote date), price (underlying price,
one usable option price), and the joined rate/dividend yield. Never trust a
vendor's IV or Greeks as ground truth — always compute our own; vendor values
are kept in separate `vendor_*` columns used only as a sanity check.
Implemented in `src/ivuq/data/schema.py`.

### 3.6 The historical-data problem — resolved in session 2

Original scope said "6 months to 1 year of option chain data." **This is not
achievable for free, retroactively.** yfinance (and every other free source)
only exposes the *current* option chain snapshot — there is no free source of
historical option chain data anywhere (that's paid specialty data: CBOE
DataShop, OptionMetrics, ORATS). Resolution, agreed with the user:
1. Pull the *entire* current chain (every strike, every listed expiry) right
   now — this alone gives a real cross-sectional smile + term structure at one
   point in time, immediately usable.
2. Re-run the pull daily going forward so real historical depth accumulates
   over the remaining project timeline, instead of trying to backfill.
Underlying *price/rate* history (not option chains) genuinely is available
years back via yfinance, so anything needing only spot/rate history isn't blocked.

**Not yet done:** actually scheduling the daily pull (e.g. Windows Task
Scheduler). Deliberately not set up automatically — it's an OS-level,
persistent change, and the user hasn't asked for it yet.

---

## 4. What is actually built and tested right now

**59/59 tests passing** (up from 42 at the end of session 1). Run
`pytest tests/ -v` from `F:\capstone` to verify (package installed editable:
`pip install -e .`).

```
src/ivuq/
├── __init__.py
├── arbitrage/                     NEW in session 2
│   ├── __init__.py
│   └── checker.py                 put-call parity, strike monotonicity,
│                                   strike convexity, calendar spread (see §4.8)
├── data/
│   ├── __init__.py
│   ├── schema.py                  column contract, REQUIRED/OPTIONAL_COLUMNS,
│   │                              option_style_for(), validate_option_chain()
│   ├── pull.py                    NEW in session 2 — full-snapshot pull +
│   │                              own-IV attachment + CSV save (see §4.6)
│   └── sources/
│       ├── __init__.py
│       ├── yahoo.py                primary loader, THREE bugs fixed this
│       │                          session (see §4.9)
│       ├── tradier.py              fallback loader — dormant, paid, untested live (§3.4)
│       └── lse_data.py             third source — field mapping CONFIRMED
│                                   live this session (see §4.7)
└── pricing/
    ├── __init__.py
    ├── black_scholes.py           price, delta/gamma/vega/theta/rho, European IV solver
    ├── binomial.py                CRR binomial tree, American + European
    ├── baw.py                     Barone-Adesi-Whaley American approximation
    ├── iv_solver.py                unified European/American implied-vol inversion
    └── comparison.py               NEW in session 2 — BS/tree/BAW comparison
                                   table on real priced data (see §4.10)

tests/
├── test_black_scholes.py     put-call parity, Hull textbook example, finite-diff
│                              Greeks checks, round-trip IV, boundary rejection
├── test_binomial.py           tree->BS convergence, American>=European,
│                              zero-dividend call==European call, deep-ITM put test
├── test_baw.py                 BAW==European when no dividend; BAW vs tree tolerance
├── test_iv_solver.py           European and American round-trip (price->IV->price)
├── test_yahoo_loader.py        mocked yfinance, schema compliance, cleaning pipeline
├── test_tradier_loader.py      mocked REST responses, schema compliance
├── test_lse_data_loader.py     mocked SDK client, field-mapping checks —
│                              UPDATED session 2 to match confirmed real field names
├── test_pull.py                NEW — snapshot aggregation, failed-expiry
│                              handling, out-of-scope rejection, CSV save
├── test_arbitrage.py           NEW — all four checks, clean + violation fixtures
└── test_comparison.py          NEW — IV-column requirement, European/American
                               reference cases, unsolvable-row skipping
```

`pyproject.toml` exists (package `ivuq`, deps: numpy/scipy/pandas/yfinance/requests;
optional extras `dev`, `fast` [numba, not installed], `lse` [lse-data, IS
installed — version 0.14.0]).

**No `.gitignore`, no git repo yet — still true in session 2, untouched.**
`__pycache__/`, `src/ivuq.egg-info/`, `.pytest_cache/` are currently untracked
junk. New in session 2: `data/raw/*.csv` also now exists (see §4.6) — a real,
if small, dataset that needs a decision (commit it, or gitignore it and treat
`data/raw/` as regenerable) before the first commit. Not decided yet.

### 4.6 `data/pull.py` — the actual live data pull (new, session 2)

Built and **run for real** against live Yahoo Finance data (not just mocked).
`pull_full_snapshot(underlying, max_expiries=None)`: pulls every currently
listed expiry, cleans each one, concatenates, attaches our own solved IV per
row via `iv_solver.implied_vol()` — rows whose market price sits outside the
no-arbitrage bracket get `implied_volatility=NaN` and `iv_solvable=False`
rather than being silently dropped. `save_snapshot()` writes to
`data/raw/{underlying}_{date}.csv`. `pull_all()` / `__main__` runs all four.

**Real results from 2026-08-13** (capped at the 15 nearest expiries per
underlying to avoid hammering Yahoo — not all listed expiries were pulled,
this was a deliberate cap, not a silent one):

| Underlying | Expiries pulled | Rows after cleaning | IV-solvable | IV-unsolvable |
|---|---|---|---|---|
| SPX | 15 | 2191 | 2191 | 0 |
| NDX | 13 | 313 | 313 | 0 |
| NVDA | 14 | 838 | 836 | 2 |
| AAPL | 14 | 712 | 705 | 7 |

Files saved: `data/raw/SPX_2026-08-13.csv`, `NDX_2026-08-13.csv`,
`NVDA_2026-08-13.csv`, `AAPL_2026-08-13.csv`.

Ticker-vs-schema-name split handled via a new `canonical_underlying` param on
`YahooMarketDataCollector`: Yahoo lists SPX/NDX chains under `^SPX`/`^NDX`
(confirmed live — `^GSPC` has zero option expiries, `^SPX`/`^NDX` have
52/44), but the schema's canonical name has no caret. Without this fix,
pulling SPX/NDX would have crashed inside `option_style_for()`.

### 4.7 `data/sources/lse_data.py` — live-confirmed, session 2 (was speculative)

Session 1 built this against guessed field names from reading their GitHub
README, with no real API response to check against. **Session 2: the user
provided a real API key in chat and it was live-tested.** Findings:

- The guessed field names were **wrong** in several places. Confirmed real
  shape of an `options()` record: `ticker` (not `symbol`), `contract_type`
  (not `type`), `last_price` (not `price`), `volume_today` (not `volume`).
  `underlying_price`, `strike`, `expiry`, `iv`, `delta`, `gamma` were guessed
  correctly. Bonus, not previously known: `theta`, `vega`, `rho` are also
  returned (not currently mapped into the shared schema — only
  `vendor_implied_volatility`/`delta`/`gamma` exist there; add
  `vendor_theta`/`vega`/`rho` columns later if useful).
- `_row_to_schema()` and `tests/test_lse_data_loader.py` were both updated to
  match the confirmed real shape.
- Confirmed live end-to-end: `get_option_chain('NVDA', max_dte=60)` returned
  3176 real rows, correctly mapped, `bid`/`ask`/`open_interest` correctly all
  `None` (the fields don't exist in the response at all — not null, absent).
- **Confirmed LSE does not cover SPX/NDX** — see §3.4. Requesting an
  uncovered underlying returns an empty list cleanly (`client.options('SPX',...)
  -> []`), not an error.
- The client is WebSocket-based even for one-shot `options()` pulls
  (`LSE(api_key=..., url='wss://data-ws.londonstrategicedge.com', ...)`).
  `client.authenticated` read `False` even after a successful `options()` call
  — did not chase this further since the actual data call worked; only
  matters if `stream()`/`subscribe_options()` behaves differently and needs
  an explicit `connect()` first. Not yet tested.
- **The API key was never written to any file** — used only as an env var for
  one session, not persisted. Do not assume it's available in a future session.

### 4.8 `arbitrage/checker.py` — new, session 2, with a real calibration bug found and fixed

Four static no-arbitrage checks: put-call parity, strike monotonicity, strike
convexity (butterfly), calendar spread. Built, tested on synthetic fixtures,
then **run against the real SPX pull** — and the first version was wrong:

- With a near-zero fixed tolerance, it flagged ~30% of SPX rows as convexity
  violations. Investigated: almost all of it was ordinary bid-ask noise
  between independently-quoted adjacent strikes, not real mispricing.
- **Fix:** every check's tolerance is now `tol + spread_multiplier * (the
  relevant quotes' own bid_ask_spread)`, falling back to the flat `tol` if
  `bid_ask_spread` isn't in the frame. This dropped SPX convexity violations
  from 686→13 and monotonicity from 23→4.
- **Open, unresolved finding, not yet explained or fixed:** NDX/NVDA/AAPL
  violation rates stayed high (parity 15-34%, convexity 18-31%) even after the
  fix. Investigated whether this was concentrated in 0-1 DTE (near-expiry,
  jumpy) contracts — **it is not** cleanly explained by DTE (e.g. NVDA's
  127-DTE bucket had a 42% convexity violation rate, higher than the 1-DTE
  bucket's 17%). Separately noticed `bid_ask_spread` is frequently exactly
  `0` (median 0 for NVDA) — meaning many rows' `market_option_price` likely
  came from a stale last-traded price (the `valid_bid_ask` fallback in
  `yahoo.py`'s `get_option_chain()`), not a live bid-ask midpoint, which would
  explain violations the spread-based tolerance can't absorb (a spread of $0
  gives zero adaptive slack even though the underlying signal is stale, not
  precise). **This was flagged to the user, not resolved** — a real avenue for
  next time: add a boolean flag for whether `market_option_price` came from a
  live midpoint vs. a stale last-trade fallback, and either exclude or
  separately weight stale rows in the arbitrage check.

### 4.9 `data/sources/yahoo.py` — three real bugs found and fixed in session 2

All found by live-testing before trusting the data, not from code review:

1. **Dividend yield was wildly wrong.** `get_dividend_yield()` returned
   0.35 (i.e. 35%) for AAPL and 0.46 (46%) for NVDA — true values are ~0.34%
   and ~0.02%. Root cause: Yahoo's `dividendYield` field changed format — it's
   now a percent-style number (`0.35` meaning "0.35%"), not a fraction, which
   silently broke the old `value/100 if value>1.0 else value` heuristic (0.35
   sits below 1.0, so it was returned unscaled). **Fix:** prefer
   `trailingAnnualDividendYield` (confirmed to already be a clean decimal
   fraction, e.g. 0.0034 for AAPL) and only fall back to `dividendYield`,
   always divided by 100. Confirmed against live AAPL/NVDA/MSFT data before
   and after the fix.
2. **The open-interest liquidity filter silently dropped 100% of some
   chains.** `clean_option_chain()`'s `min_oi=20` default is a hard filter,
   and live-tested `open_interest` is essentially unpopulated right now across
   the board (0/392 rows for a real SPX pull, 2/189 for NVDA — not an
   SPX-specific gap). **Fix:** changed the default to `min_oi=0` (not
   enforced); OI is documented as optional in the schema for exactly this
   reason, and liquidity filtering now leans on volume + spread instead.
3. **Canonical-name mismatch for indices** — see §4.6. Added
   `canonical_underlying` param to `YahooMarketDataCollector`.

### 4.10 `pricing/comparison.py` — new, session 2, run against real data

`compare_pricers(df)` requires `implied_volatility` already attached (from
`pull.py`), reprices every row with BS/tree/BAW as applicable, and adds
diff columns. Run against the real pulled data:
- SPX/NDX (European): BS vs tree agree to ~1 cent mean, ~15-40 cents max
  (tree discretization noise; expected, not a concern).
- NVDA/AAPL (American): BAW vs tree mean gap ~1-1.5 cents, max ~20-28 cents —
  consistent with BAW being a fast approximation to the tree, not exact.

### 4.11 The BAW bug (session 1, kept for reference — worth knowing about if pricing numbers ever look wrong again)

While building `baw.py`, the first implementation was **wrong by ~35x on the
early-exercise premium** for near-the-money, long-dated calls with modest
dividend yield (e.g. S=K=100, r=5%, q=3%, σ=30%, T=1y: computed critical
exercise price ~613, true value — confirmed via QuantLib — is ~213.5).

Root cause: a discount-factor term was dropped from the 1987 paper's `Q`
formula. The correct form needs `4*M/K_disc` inside the square root, where
`K_disc = 1 - exp(-r*T)`, not just `4*M`. Found by installing QuantLib-Python
and cross-checking every price against its independent
`BaroneAdesiWhaleyApproximationEngine`, then bisecting for the actual
critical price to localize the discrepancy.

**Lesson for future debugging:** when a pricing formula looks textbook-correct
but produces implausible numbers, don't trust a re-derivation against itself —
get an independent, trusted reference implementation (QuantLib is good for
this) and bisect. `QuantLib` and `py_vollib` are `pip install`-ed into this
environment for that purpose; they are **not** project dependencies (not in
`pyproject.toml`).

### 4.12 Numerical note: American IV solver's lower vol bound

`iv_solver.implied_vol()` defaults to `lo=1e-2` (1% vol), not near-zero. At
very low sigma, the CRR tree's risk-neutral probability `p` can fall outside
(0,1) for realistic step counts — a discretization artifact, not a bug. No
real option prices below 1% IV anyway, so this costs nothing.

---

## 5. Known doc inconsistencies — still not cleaned up

Unchanged since session 1 — **no doc cleanup happened in session 2**, all the
work was code + a live data pull. Still stale: `README.md`, `ADDON.md`,
`PROJECT_FLOW.md`, `flow.md`, and scattered spots in `PROJECT_PLAN.md` /
`BUILD_PLAN.md` (India VIX, NSE holiday calendar, the "reduce to two groups"
emergency lever, cross-market transfer-learning descriptions, bhavcopy loader
references). Do a dedicated cleanup pass across all six markdown files before
anyone reads them cold — don't do it piecemeal while also touching code.

Also now stale in a new way: `BUILD_PLAN.md`/`tradier.py`'s docstring still
frame Tradier as a free sandbox fallback — this is wrong per §3.4 and should
be corrected in the same cleanup pass.

---

## 6. Explicit user instructions (cumulative — don't relitigate these)

From session 1:
- **Do not build the conformal-prediction sandbox** (Phase 13) — explicitly
  excluded. Still true; not cancelled, just deferred.
- **Keep the code architecture simple** — plain functions and small
  dataclasses, not base-class/interface abstractions, until there's an actual
  second implementation that needs swapping in.
- **Don't take destructive/shared-state actions without asking first** — a
  large chunk of the repo was scaffolded once before the user had confirmed
  they wanted that yet, and had to be reverted. Since then: confirm scope
  before writing, especially anything touching planning docs or git.
- **`git init` has deliberately not been done** — user confirmed pushing to
  GitHub is fine (no secrets in code) but has not asked me to actually run
  `git init`/first commit yet.

From session 2:
- **Tradier is paid, not free — do not treat it as an active fallback** or
  re-propose it without the user re-confirming pricing has changed.
- **CBOE/Alpha Vantage/Polygon were considered and explicitly parked** — rate
  limits judged not worth the integration effort. Yahoo + LSE only, no
  fallback slot, accepted gap.
- **"Let me know first in chat" pattern reinforced repeatedly** — before
  pulling real data, before assuming a design choice (e.g. the historical-data
  problem in §3.6) — explain in chat and get a go-ahead before executing,
  especially for anything with real-world side effects (live API calls,
  file writes outside a sandbox).
- **The API key given in chat must never be written to a file or echoed
  back** — handled correctly this session (used transiently via env var only).
  Apply the same rule to any future credential shared in chat.
- **PINN/modeling work is explicitly halted for now** (see §7) — the user
  wants to push to GitHub first, then decide on more features, before any
  N0/N1/N2 math or code starts. Do not start on this unprompted.

---

## 7. Natural next steps (not started, no commitment implied — session 2 update)

The user explicitly said: **push to GitHub first, then decide on more
features — halt everything else, including any PINN/modeling work, until
then.** So the actual next action, when the user is ready, is:

1. `.gitignore` + decide what to do with `data/raw/*.csv` (commit the real
   pull, or gitignore `data/raw/` as regenerable) + `git init` + first commit
   — still blocked on explicit user go-ahead, now doubly so per the halt above.
2. Whatever "more features" turns out to mean — not yet specified by the user.

Deliberately paused, not cancelled, pending the above:
3. Doc consistency sweep (§5).
4. The arbitrage-checker open finding from §4.8 (stale-quote vs live-quote
   flag) — worth revisiting before trusting the violation-rate numbers for
   anything in the paper.
5. Daily-snapshot scheduling (§3.6) — script exists (`pull.py`), OS-level
   scheduling does not.
6. Model roadmap, discussed but explicitly not started: `BUILD_PLAN.md` lists
   N0 (control net) through N12, tiered Required (N0-N3) / Recommended
   (N4-N8, includes neural ODE N5 and neural SDE N7) / Optional (N9-N11) /
   Cuttable (N12). Next required models after pricing+data are N0 (control)
   and N1 (European PINN), per the doc's own sequencing and its "B8 PINN...B
   (equation from A)" framing — i.e. the user is expected to hand precise
   math (loss formulation, boundary/terminal conditions, how σ enters the PDE
   residual, input parametrization) for Claude to implement. N2 (American
   free-boundary PINN) is the flagship/centerpiece contribution and the
   hardest math (a complementarity/free-boundary system, not one PDE). The
   user was offered the choice of starting with N0/N1 (simpler, matches
   sequencing) or jumping straight to N2 (harder, higher-value) and chose to
   **halt on this entirely** until after the GitHub push and further feature
   decisions — revisit this section's framing when that happens, don't assume
   the choice below still needs asking fresh.
