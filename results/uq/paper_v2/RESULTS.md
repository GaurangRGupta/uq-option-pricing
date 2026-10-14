# v2 results: N2 bias fix, P2 and P4 rerun

Times are IST, 12 Oct 2026. Protocol: PROTOCOL.md, written before any v2
coverage was computed. The cause is analysed in DIAGNOSIS.md and the code
diff is in CODE_CHANGES.md. Every number below comes from the CSV named
with it. `v1_vs_v2.csv` holds all of them side by side.

## Cause (confirmed) and fix
- **Cause:** the soft LCP penalty used the product L[u](u - h). It barely
  penalises L[u] > 0 where u - h is small, so training converged to a
  supersolution above the true price. The v1 models have L[u] of about 0.042
  at 96.7% of continuation points; it should be 0.
- **Suspected m = 0 boundary:** refuted as the cause. It points the wrong
  way (down, not up), and fixing it alone moves the bias by only about
  $0.09.
- **Fix:** min-form penalty min(L[u], u - h), complementarity weight 10, and
  the American value 1 at m = 0. Everything else is v1.
- **Not retrained:** N0 and N1, because the European code was not touched.
  Their re-evaluated numbers match v1 to within 3e-5.

## Environment
The box had been reset. I rebuilt `.venv` with the pinned versions from
`../paper/env_versions.txt` (Python 3.12, torch 2.14.1+cpu, numpy 2.5.3).
Hardware is the same: 8 vCPU Xeon, CPU only.

## Tests
`pytest_full_suite.txt`:
- After the code change: 121 passed, 1 warning (4 min 24 s).
- After adding `tests/test_pinn_american_lcp_form.py` (3 tests): full suite
  124 passed, 1 warning (4 min 09 s).
- Final check of the PINN tests on the final files: 15 passed.

## Wall time
| Step | Wall time | Source |
|---|---|---|
| Diagnosis step 1 (v1 models, no training) | about 9 s | `scripts/v2_diag_v1models.py` |
| Diagnosis batch 1 (8 runs) | 01:35:17 to 01:38:22 | `diagnosis/logs/batch1_*.txt` |
| Diagnosis batch 2 (8 runs) | 01:40:47 to 01:43:58 | `diagnosis/logs/batch2_*.txt` |
| 10 N2 trainings (8 parallel) | 01:45:44 to 01:51:04 (5 min 20 s) | `logs/train_start.txt`, `logs/train_end.txt` |
| per N2 run | 165.1 +/- 9.8 s (v1 229.1 s) | `p1p2/p1p2_model_diagnostics_summary.csv` |
| P1/P2 conformal | 32.8 s | `p1p2/p1p2_meta.json` |
| P4 | 16.5 s | `p4/p4_meta.json` |

## N2 point accuracy (pools, 10 seeds)
Sources: `p1p2/p1p2_model_diagnostics_summary.csv` and
`p1p2/p1p2_signed_error_summary.csv`.

| | v1 | v2 |
|---|---|---|
| MAE | $3.13 | **$0.29** (+/- 0.19) |
| RMSE | $3.48 | $0.32 |
| Max error | $6.89 | $0.84 |
| Mean signed error (bias) | +$3.13 | **+$0.22** (+/- 0.24) |
| Share above CRR | 99.6% | **83.7%** |
| Spearman, abs(L[u]) vs abs(err) | -0.01 | 0.01 |
| Spearman, abs(LCP residual) vs abs(err) | 0.77 | **0.38** |

- The bias is lower in all 10 seeds. v1 ranged from +$2.45 to +$3.79; v2
  ranges from -$0.13 to +$0.82, and seed 6 is the outlier at +$0.82.
- The decision rule in PROTOCOL.md is met: bias below $1 and lower in at
  least 8 of 10 seeds.
- The fix narrows the bias but does not remove it. The deep out-of-the-money,
  long-tenor bucket (m > 1.1, tau >= 0.4) is still +$0.35, with 99.7% of its
  points above CRR.
- For comparison, N1 has MAE $0.25 and N0 $4.42, both unchanged.

## P2 conformal (N2, n_cal = 800; `p1p2/p1p2_summary.csv`)
| alpha | Method | Cov. v1 | Cov. v2 | Width v1 | Width v2 | Worst bucket v1 | Worst bucket v2 |
|---|---|---|---|---|---|---|---|
| 0.10 | Split | 0.900 | 0.901 | $10.05 | **$0.93** | 0.71 | 0.66 |
| 0.10 | Norm. abs(L[u]) | 0.900 | 0.901 | $11.89 | $1.97 | 0.76 | 0.65 |
| 0.10 | Norm. LCP | 0.900 | 0.900 | $9.01 | $1.13 | 0.71 | 0.64 |
| 0.10 | Mondrian | 0.904 | 0.904 | $8.67 | **$0.85** | 0.86 | 0.86 |
| 0.05 | Split | 0.950 | 0.951 | $11.08 | $1.04 | 0.82 | 0.78 |
| 0.05 | Norm. abs(L[u]) | 0.950 | 0.950 | $12.79 | $2.24 | 0.85 | 0.81 |
| 0.05 | Norm. LCP | 0.950 | 0.951 | $9.96 | $1.30 | 0.81 | 0.80 |
| 0.05 | Mondrian | 0.954 | 0.954 | $9.15 | $0.92 | 0.92 | 0.92 |

What changes in the reading:
- **Coverage:** marginal coverage stays nominal for every score (finite-sample
  guarantee). Bands are about 11x narrower ($10.05 to $0.93): the bias was
  costing width.
- **Normalised LCP:** it **no longer narrows the band.** It is wider than
  split ($1.13 vs $0.93) and its worst bucket is lower (0.64 vs 0.66). Its
  rank correlation fell from 0.77 to 0.38. In v1, most of the error was the
  supersolution excess, which L[u] > 0 signalled. In v2 the remaining error
  is smaller and less tied to the residual.
- **LCP vs raw residual:** the LCP residual still beats abs(L[u]), with
  correlation 0.38 vs 0.01 and width $1.13 vs $1.97 at equal coverage. So
  "the residual must respect complementarity" still holds against the raw
  residual. But in v2 no residual normalisation beats plain split on width.
- **Mondrian:** still the best. It gives the narrowest bands and every bucket
  sits between 0.901 and 0.906 on average (v1: 0.902 to 0.910).
- **Split's worst bucket** is now m < 0.9 with tau < 0.4 (0.69). In v1 it was
  m < 0.9 with tau >= 0.4 (0.71).
- N1 and N0 rows are identical to v1.

## P4 (N2, tol 1e-6, 10 seeds; `p4/p4_summary.csv`)
| Series | Monot. v1 to v2 | Convex v1 to v2 | Calendar v1 to v2 | Price < 0 v1 to v2 |
|---|---|---|---|---|
| N2 point price | 0.010 to 0.039 | 0.260 to **0.121** | 0.000 to **0.168** | 0.00 to 0.02 |
| N2 split, lower | 0.059 to 0.053 | 0.260 to 0.121 | 0.000 to 0.168 | 0.36 to 0.20 |
| N2 split, upper | 0.001 to 0.033 | 0.260 to 0.121 | 0.000 to 0.168 | 0.00 to 0.00 |
| N2 norm. LCP, lower | 0.131 to 0.113 | 0.388 to 0.266 | 0.042 to 0.229 | 0.35 to 0.20 |
| N2 norm. LCP, upper | 0.028 to 0.042 | 0.399 to 0.226 | 0.077 to 0.240 | 0.00 to 0.00 |
| N2 Mondrian, lower | 0.032 to 0.051 | 0.321 to 0.145 | 0.110 to 0.164 | 0.32 to 0.18 |
| N2 Mondrian, upper | 0.016 to 0.032 | 0.322 to 0.148 | 0.000 to 0.180 | 0.00 to 0.00 |

The fix trades one arbitrage error for another:
- **Convexity:** violations halve (26% to 12%, about the N1 level).
- **Calendar:** **violations appear, 17% of checkable pairs (v1: 0%).** v1's
  surplus L[u] > 0 made u rise in tau, which hid calendar errors.
- **Point prices:** 2.4% are now negative (v1: 0%).
- **Lower edges:** about a fifth are negative, down from about a third.
- **Obstacle u >= h:** less strictly met than in v1 (DIAGNOSIS.md).
- No repair was applied. This is a real trade-off to report, not hide.

## Figures (`figures/`, same names as v1)
- `fig_p2_residual_vs_error`: seed 0. Median abs(L[u]) is 0.0071 (v1: 0.057)
  and median error is $0.26 (v1: $2.43). The LCP residual tracks the error
  mainly along the exercise-region branch, where it equals the obstacle gap.
- `fig_p2_bucket_coverage_a10`: only Mondrian sits at target in every bucket.
- `fig_p2_interval_slice_tau050`: bands are tight around CRR. Out of the money,
  the lower edges still go below zero.
- `fig_p4_n2_arbitrage`, `fig_p1p2_marginal_coverage`, and the P1/P3 figures
  were regenerated. The P3 inputs are copied unchanged from v1.

## Preview tables
`tables_preview/` holds the paper tables rebuilt from v2. It is a copy of
`paper_ieee/make_tables.py` pointed at paper_v2. Tables I and IV and
values.tex are unchanged. Only the N2 rows of Tables II, III and V change.
`paper_ieee/` was not edited apart from `paper_ieee/v2_update_plan.md`.

## Deviations
None from PROTOCOL.md. Note: the N0/N1 re-evaluation differs from v1 by up
to 2.6e-5 in some metrics. This is float noise from a fresh environment with
the same versions.
