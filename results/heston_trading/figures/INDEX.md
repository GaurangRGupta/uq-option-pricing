# Figure index: what each picture shows and what it says

All figures are built by `PYTHONPATH=src python scripts/make_report_figures.py` from the study results. P&L is in dollars per one-contract
trade, delta-hedged, horizon 3 trading days, costs included (0.65 per contract per side, 1 bp on stock traded). "Development" is
2025-10-16 to 2026-04-30, "test" is 2026-05-01 to 2026-10-21; settings (alpha 0.10, window 20, horizon 3) were fixed before the test period was
looked at. "Same close" means filled at the quote of the signal day (optimistic), "next close" at the following day's quote.

## The market (implied volatility)
| File | Shows | Reading |
|---|---|---|
| `iv_surface_time_{SPY,AAPL,NVDA}.png` | implied vol over moneyness and calendar date at a 30-day constant maturity | the smile and level move over the year; NVDA is noisier and higher |
| `iv_surface_dolthub_2026-10-21_*.png`, `iv_surface_yahoo_live_2026-10-23_*.png` | implied vol over moneyness and days to expiry for one date | term structure and skew of the day; the Yahoo chain is far denser |

## The model against the market
| File | Shows | Reading |
|---|---|---|
| `network_accuracy.png` | networks against fresh exact-solver prices | about 4 to 7 cents per $100 strike (anchored networks, see the paper) |
| `smile_fit_dolthub_2026-10-21_*.png` | bid-ask implied vols, model theo and conformal band, by expiry | the theo sits inside the spread for most strikes; the band is wide in vol terms where vega is small |
| `error_map_heston.png`, `error_map_bs_flat.png` | median error in units of the error scale c over moneyness x tenor | Heston misses most on out-of-the-money puts (S/K above 1.0), where the smile is steepest and no jumps are modelled |
| `params_time_single_network.png` | daily calibrated parameters, edge-of-box points marked | kappa and xi sit on the box edge on many days: the five parameters are not well identified from one day's puts |

## The band (conformal) and the trades
| File | Shows | Reading |
|---|---|---|
| `conformal_coverage.png` | realised miss rate of the band by underlying, bucket and month | about 10 to 11% overall against the 10% target; months range from 4% to 15%, so coverage is right on average but not stable |
| `ablation_band.png` | each theo traded with and without the band, development and test, same and next close | the band is what turns the Heston theos from losing (any edge after the spread: all negative in test) into profitable at the same close (single network +$9.5k, coupled +$8.3k); BS flat vol with the band also profits (+$6.3k); the exact solver does not |
| `equity_lag0.png`, `equity_lag1.png` | cumulative P&L | at the same close the band curves are flat or down until March 2026, then rise; at the next close every curve falls |
| `lag_decay.png` | test-period totals, same close against next close | the profit disappears when the fill is one day later |
| `model_vs_random.png` | model totals against 30 random-signal runs with the same trade counts | random trades lose heavily (spread and costs), so the model's same-close result is not luck of the cost structure |
| `breakdown_{pinn,cpl1000}_lag0.png` | mean P&L per trade by underlying, side, tenor, moneyness | in the test period every underlying is positive (SPY has only 23 trades), buys earn more than sells, and in-the-money puts (S/K below 0.95) earn most (about $150 per trade) while out-of-the-money puts (S/K above 1.05) lose (about $45 per trade); in development the same cells are near zero |
| `cost_sensitivity_{pinn,cpl1000}.png` | mean P&L per trade as commission and hedge cost rise | test-period edge survives 4x hedge costs and $2.60 commissions; development edge does not |
| `trade_example_pinn_{best,worst}.png` | one trade from entry to exit with the theo and band | what a signal and its outcome look like |

## Failures and limits
| File | Shows | Reading |
|---|---|---|
| `failure_{pinn,cpl1000}_lag0.png`, `failure_pinn_lag1.png` | P&L distribution, monthly P&L, P&L against signal strength, where the dollars go | median trade is about zero; the profit is a fat right tail; months in Dec 2025 to Feb 2026 lose; larger signals earn more |
| `gap_closure.png` | next-day hedged mid change against today's (theo - mid) gap | the market closes only about 11% of a Heston gap by the next close (BS flat vol 9%): quote dislocations here are mostly noise that the next close does not fix, which is why next-close fills lose |
| `live_shift.png`, `live_tags.png`, `signal_map_yahoo_*.png` | the live Yahoo chain against the DoltHub history | the history is a sparse subset (SPY about 85 quotes a day against 1,235 live); on the dense chain the networks (narrow parameter box) fit worse than the exact solver (wide box) and the band covers 74 to 78% instead of 90%; live signals form contiguous strike blocks, which is model misfit, not isolated mispricings |

## Honest summary for a presentation
1. The networks reproduce the exact solver to a few cents per $100 strike, so they are usable as a fast theo.
2. The conformal band does what it promises on average (coverage near 90%) and it is the component that makes trading the theo viable at all.
3. The profit appears at the same-close fill, mostly in the second half of the sample, and is matched by plain Black-Scholes with the band: it is a property of end-of-day quote noise plus the band, not of Heston dynamics.
4. At the next-close fill nothing makes money: the gap is not closed by the next day (k about 0.1).
5. On a different, denser quote source (live Yahoo, one day) the band calibrated on the old source under-covers, and the narrow-box networks fit worse than the exact solver. Transfer needs recalibration on the new source and, for the networks, a wider training box.
