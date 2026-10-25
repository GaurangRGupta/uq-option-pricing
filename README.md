# ivuq — Option Pricing with Uncertainty

**Status: still being built.** This is an active capstone project, not a
finished product. This README shows what's done so far and what's coming next.

## What is this project about?

When you price an option, most models (like Black-Scholes) just give you one
number: "this option is worth $4.32." But that number is never exactly right
— real markets are messy, volatility isn't constant, and thin/illiquid
options can be priced with a lot of uncertainty.

Our goal is to build a system that doesn't just predict a price, but also
tells you *how confident it is* — a range instead of a single number, backed
by real statistical guarantees instead of just a gut feeling. And that range
has to make sense: it can't accidentally suggest a "free money" trade that
breaks basic no-arbitrage rules.

## What makes this project different

A few things we're aiming for that most simple "ML for option pricing"
projects don't do:

1. **Learn the early-exercise boundary for American options, with a
   confidence band around it.** There's no formula for this boundary — it
   has to be learned, and we want to know how sure we are about it.
2. **Balance "honest uncertainty" against "no free money."** A model can give
   very safe, wide confidence ranges, but if those ranges are wide enough to
   imply an arbitrage opportunity, that's not actually useful. We're trying to
   measure and balance both at once.
3. **Use the pricing equations themselves to judge difficulty.** Neural
   networks that also try to satisfy the Black-Scholes equation can tell you,
   at any point, how badly they're breaking that equation. We want to reuse
   that number as a free signal for "how hard is this option to price," instead
   of training a whole separate model just to guess that.
4. **Show why splitting data randomly is a mistake for market data.** If you
   split by date vs. randomly, you get two very different-looking results.
   We want to actually show that side by side, not just claim it.

## What's actually done so far

**The pricing math (the foundation everything else builds on):**
- Black-Scholes pricing, all the Greeks, and a solver that backs out
  volatility from a price
- A binomial tree model for pricing American options (options you can
  exercise early)
- A faster approximate American-option pricing method (Barone-Adesi-Whaley)
- All three of these were checked against QuantLib (a widely-trusted
  open-source pricing library) to make sure the numbers are actually correct

**Getting real market data:**
- A loader that pulls live option prices from Yahoo Finance, cleans up bad
  quotes, and checks that the data makes sense
- A second data source (London Strategic Edge) as backup — though we found
  out it only covers individual stocks, not index options like SPX
- We're focusing on 4 things: SPX and NDX (index options, which are simpler
  "European-style") and NVDA and AAPL (stock options, "American-style," which
  allow early exercise)

**Making sure the numbers make sense:**
- A checker that looks for basic pricing inconsistencies in real market data
  (like a call option that should never cost less than a shorter-dated one
  at the same strike, but sometimes does due to quote noise)
- A comparison tool that checks our three pricing methods against each other
  on real data we pulled

**Tests:** 142 automated tests, all passing, so we can keep changing things
without accidentally breaking what already works.

## Heston American-put theo and a hedged backtest (October 2026)

A neural network that prices American puts under the Heston stochastic-variance
model for any parameter set, used the way a trading desk uses a theo: fit it to
each day's real option quotes, compare the market with the theo, and trade only
when the gap is bigger than a conformal band.

- `src/ivuq/pricing/heston_fd.py`: an exact finite-difference solver (the reference).
- `src/ivuq/pinn/heston_cond.py`, `heston_cond_coupled.py`: the parameter-conditioned
  networks (single network with a Black-Scholes control variate, and a coupled
  solution + exercise-boundary pair). The networks used for trading are anchored
  to exact-solver labels, and the pure-physics versions are kept as an ablation.
- `src/ivuq/calibration/heston_calib.py`, `src/ivuq/trading/`: daily walk-forward
  calibration, past-only adaptive conformal band, delta-hedged take-the-quote
  backtest, random-signal controls.
- `scripts/pull_dolthub.py` rebuilds the quote set from DoltHub (AAPL, NVDA, SPY,
  CC BY-SA 4.0, not committed). `scripts/run_heston_theo.py`, `run_heston_study.py`,
  `run_oos_test.py`, `make_signals.py` produce the results in `results/heston_trading/`.
- `paper_heston/` holds the write-up; `paper_heston/make_tables.py` builds every table from the results.

Result in one line: the band holds its coverage and the networks match the exact
solver to about five cents per $100 of strike, but filled at the next close no
theo (Heston or Black-Scholes) makes money, so there is no tradable edge on
end-of-day data. The same-close profit is matched by a plain Black-Scholes theo.

## What's coming next

Everything above is the foundation. Here's the plan from here:

- **A baseline neural network** — a plain model with no extra physics, so we
  have something fair to compare our fancier models against.
- **A neural network that also has to obey the Black-Scholes equation** (for
  European options, the simpler case) — trained to match real prices *and*
  to not violate the pricing math.
- **The big one: a neural network for American options that learns the
  early-exercise boundary itself.** This is the main thing we want to show
  off — nobody has a formula for this boundary, so a model that learns it
  (with a confidence range) is genuinely new.
- **The uncertainty layer** — this is where we add the confidence
  ranges/bands on top of whatever model we're using, using a statistical
  technique called conformal prediction, plus the "use the equation to judge
  difficulty" idea from above.
- **If we have time:** a few extra models — ones that are mathematically
  built to never allow certain types of arbitrage, and models that try to
  learn how the market moves over time instead of just fitting a snapshot.

## Try it yourself

```bash
pip install -e .
pytest tests/ -v
python -m ivuq.data.pull        # pulls a live snapshot for SPX/NDX/NVDA/AAPL
```

## Built with

Python, NumPy, SciPy, pandas, yfinance. QuantLib was used during development
just to double-check our pricing math — it's not needed to actually run the project.
