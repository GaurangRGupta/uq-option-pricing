# Diagnosis of the v1 N2 upward bias (+$3.13, 99.6% of pool points above CRR)

All times are IST. Scripts: `scripts/v2_diag_v1models.py` (step 1, which
inspects the v1 models without retraining) and `scripts/v2_diag_experiment.py`
(step 2, single trainings). Scoring uses a separate diagnostic grid,
`diagnosis/diag_grid_crr500.csv`: 53 x 20 points, m in [0.3, 1.6], tau in
[0.05, 1], with 500-step CRR labels. It is not a calibration/test pool. Dollar
figures use K = 100. "Eval region" means m in [0.6, 1.4], the pool domain.

## Step 1: what the v1 models do (10 seeds, `diagnosis/v1_model_diag.csv`)

| Quantity (mean over 10 seeds) | Value | Reading |
|---|---|---|
| Signed error, whole grid | +$2.63, 98.6% of points above CRR | Reproduces the pool finding. |
| Exercise region (CRR = h) | +$2.14 | Too high even where the answer is just h. |
| Continuation, m < 1 | +$2.96 | |
| Out of the money, m >= 1 | +$2.83 | |
| L[u] in the continuation region | mean 0.042, positive at 96.7% of points | **True solution has L[u] = 0 here.** The net is a strict supersolution. |
| u(0, tau) | mean 0.985; 0.971 at tau = 1 (European e^{-r} = 0.970; American value = 1) | It learned the European edge value it was given. |
| abs(u(m_max, tau)) | at most 0.006 | Upper edge fine. |
| Terminal fit abs(u(m,0) - h) | max 0.015, mean signed +$0.10 | Terminal fine. |
| Fraction with u < h - 1e-4 | 1.25% | Obstacle is enforced. The bias is upward, not below h. |
| Final loss terms (seed 0) | complementarity 1.7e-6, ineq 1.9e-7, terminal 3.0e-5 | All tiny: the optimiser thinks it is done. |

**Mechanism.** In the continuation region, only the complementarity term
w (L[u] (u - h))^2 penalises L[u] > 0. The inequality term relu(-L[u])
penalises only negative L. In v1 the penalty is multiplied by (u - h)^2. Out
of the money, h = 0 and u is about 0.01 to 0.05, so the effective weight on
L[u]^2 is only about 1e-4 to 2.5e-3. The terminal weight is 3. A smooth
supersolution with L[u] around 0.04 therefore costs about
(0.04 x 0.03)^2 = 1.4e-6. That is exactly the size of the v1 final
complementarity loss. A positive L[u] carried through tau up to 1 adds
roughly 0.04 x 0.5 = 0.02 in u, which is +$2, matching the observed +$2-3.
The product form of the LCP penalty does not separate the true price (the
smallest supersolution above h) from higher supersolutions.

## Step 2: isolating experiments (6000 epochs, the v1 config except the stated change)

Batch 1 ran 01:35-01:38 IST and batch 2 ran 01:40-01:44 IST, 12 Oct 2026, 8 runs
in parallel at 1 thread each, about 165 s per run. Rows are in
`diagnosis/experiments_all.csv`.

| Run | Change vs v1 | Seeds | MAE eval | Bias eval | % above | L[u] cont. | % u < h |
|---|---|---|---|---|---|---|---|
| E0 | none (v1 reproduced) | 0 | $2.51 | +$2.50 | 98.2% | 0.043 | 3.1% |
| E1 | m=0 boundary = 1 (American) | 0 | $2.41 | +$2.41 | 98.2% | 0.043 | 4.2% |
| E2 | min-form LCP min(L[u], u-h) | 0, 1 | $0.50 | +$0.46 | 89.9% | 0.0056 | 19% |
| E3 | E2 + E1 | 0, 1, 2 | $0.40 | +$0.31 | 81.6% | 0.0051 | 29% |
| E4 | product form, complementarity weight x100 | 0 | $0.86 | +$0.84 | 93.8% | 0.014 | 16% |
| E5 | curriculum (points near free boundary) | 0 | $2.53 | +$2.53 | 98.8% | 0.040 | 1.9% |
| E6 | terminal and boundary weights x10 | 0 | $2.49 | +$2.47 | 95.3% | 0.054 | 9.0% |
| **F1** | **E3 + complementarity weight 10** | **0, 1, 2** | **$0.215** | **+$0.065** | **68.9%** | **0.0026** | **31%** |
| F2 | F1 + obstacle weight 10 | 0 | $0.35 | +$0.30 | 91.1% | 0.0035 | 18% |
| F4 | E3 + complementarity weight 100 | 0 | $0.39 | +$0.22 | 68.6% | 0.0020 | 39% |
| F5 | E3 + curriculum | 0 | $0.83 | +$0.82 | 98.9% | 0.0054 | 13% |

F1 per seed: MAE $0.268 / $0.213 / $0.163; bias +$0.217 / -$0.130 / +$0.107.

## Verdict on each candidate

- **Suspected cause, the m = 0 boundary (European e^{-r tau} instead of 1):
  refuted as the cause.** The direction is wrong: e^{-r tau} < 1 pulls u down,
  not up. Changing it alone moves the bias from +$2.50 to +$2.41 (E1). It is
  still a real inconsistency, since it conflicts with u >= h = 1 at m = 0. It
  helps a little once the main cause is fixed (E2 +$0.46 to E3 +$0.31), so v2
  keeps it.
- **Confirmed cause: the LCP penalty form and weight.** The product
  L[u](u - h) barely penalises L[u] > 0 where u - h is small. Switching to
  the min form cuts the bias about 5x (E2). A large weight on the product
  helps only partly (E4). The min form with weight 10 (F1) removes most of
  the remaining bias.
- Terminal condition: not the cause. It fits to 0.015 in u, and weighting it
  10x (E6) changes nothing.
- Upper m boundary: not the cause. abs(u(3, tau)) is at most 0.006.
- Sampling domain vs eval domain: uniform m in [0, 3] puts 27% of the
  interior points in [0.6, 1.4]. That costs resolution, but it does not
  create a one-sided bias. Not tested further.
- Collocation density near the free boundary: not the cause. Curriculum
  sampling (E5, F5) does not help and hurts once the min form is used.
- The exercise constraint u >= h: it was enforced in v1 (1.25% of points
  below h). **The fix trades this away a little.** With F1, 31% of grid
  points sit below h by more than 1e-4 (more than $0.01). These are small
  undershoots near and inside the exercise region. P4 reports the effect.
  A heavier obstacle weight (F2) reduces the undershoot but brings back
  bias. That trade-off is not resolved here.
- Loss balancing: matters only through the complementarity weight (E4, F1
  and F4). The terminal and boundary weights do not matter (E6).

## Choice for v2 (fixed before protocol runs; see PROTOCOL.md)
`lcp_form="min"`, `american_low_boundary=True`, `lambda_complementarity=10.0`
(F1). This was the lowest mean validation MAE among the configurations tried.
No configuration was tried on the calibration/test pools.
