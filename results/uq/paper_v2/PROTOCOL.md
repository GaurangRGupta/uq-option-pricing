# Pre-registered protocol for v2 (N2 bias fix), P2 and P4 rerun

Written 2026-10-12 01:45 IST. This is BEFORE any v2 N2 model was
trained at protocol scale and BEFORE any v2 calibration/test coverage was
computed. The only v2 evidence used to choose the fix is in DIAGNOSIS.md: single
N2 trainings (seeds 0-2) scored on a separate diagnostic grid
(diagnosis/diag_grid_crr500.csv: a fixed 53 x 20 grid, m in [0.3, 1.6], tau
in [0.05, 1], CRR-500 labels). That grid is not one of the calibration/test
pools and was never used for conformal calibration or test.

## Identical to v1 (results/uq/paper/PROTOCOL.md)
Everything below follows v1 exactly:
- Market: GBM, r = 0.03, q = 0, sigma = 0.20, K = 100.
- Network, optimiser, schedule, epochs, point counts and domain: the
  AmericanPINNConfig defaults (4 x 64 tanh, 6000 Adam epochs, lr 2e-3 halved
  every 2000, 3000/400/200 points, m_max 3, tau_max 1), weighting "fixed".
- Training seeds 0..9, one torch thread per run, 8 runs in parallel.
- Pools: the v1 pools, unchanged (copied to paper_v2/pools; numpy
  default_rng(100000 + s), 4000 points, m ~ U[0.6, 1.4], tau ~ U[0.05, 1]).
- Splits: default_rng(200000 + 100 s + r), r = 0..19. n_cal in {400, 800},
  n_test = 2000, alpha in {0.10, 0.05}.
- Methods: split (kappa = 0); normalised |L[u]| (scale_kappa); normalised
  LCP residual |min(L[u], u - h)| (N2); Mondrian 3 x 2 buckets
  {m < 0.9, 0.9-1.1, m > 1.1} x {tau < 0.4, tau >= 0.4}.
- Summaries: average over splits within a seed, then mean and std over seeds.
- P4: the v1 script and grid, unchanged, applied to v2 N2.

## The only change (N2 training)
AmericanPINNConfig(option_type="put", weighting_scheme="fixed", seed=s,
  lcp_form="min", american_low_boundary=True, lambda_complementarity=10.0)
1. The complementarity penalty is the min-form LCP residual min(L[u], u - h).
   v1 used the product L[u](u - h).
2. The m = 0 boundary value of the put is the American value 1. v1 used the
   European value exp(-r tau).
3. The weight on the complementarity term is 10. v1 used 1.

Selection rule, fixed before the protocol runs: among the configurations
tried in DIAGNOSIS.md, take the one with the lowest mean absolute error on
the diagnostic grid, averaged over the seeds tried (F1: $0.215 over seeds
0-2). No further tuning.

N0 and N1 are NOT retrained. The code change is opt-in and defaults to v1
behaviour, so EuropeanPINN code paths are untouched. P1 is rerun only to
fill the same summary files: the v1 N0/N1 models are copied unchanged, so
their numbers must match v1.

## Outputs (results/uq/paper_v2/)
models/, pools/ (copy), p1p2/, p4/, figures/, RESULTS.md, CODE_CHANGES.md,
DIAGNOSIS.md, v1_vs_v2.csv.

## Decision rule
If the 10-seed mean N2 bias on the pools is not clearly smaller than v1's
$3.13, v1 stays as the paper's numbers. Clearly smaller means below $1.00
and lower in at least 8 of 10 seeds. In every case v2 is reported as it
comes out. Coverage results do not decide v1 versus v2.
