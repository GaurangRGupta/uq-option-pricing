# Table I baseline rerun with the v2 N2 (12 Oct 2026, IST)

Script: `scripts/v2_baseline_pinn_row.py`.
- **Harness:** `ivuq.baselines.experiments.run_regime`, unchanged. Same 25 x 9
  test grid (m 0.6-1.8, tau 0.1-0.9), same regimes (full, extrapolate,
  noisy) and the same scoring.
- **PINN:** trained as `scripts/run_baseline_comparison.py` does
  (AmericanPINNConfig defaults, weighting "fixed", seed 0, one model), plus
  the v2 fix: lcp_form "min", American m=0 boundary, complementarity weight 10.

## Reproduction check
`--variant v1` retrains the old PINN in this environment.
`check_v1_pinn_row_*.csv` reproduces the original row in
results/baselines/american_put_*.csv: MAE $2.3129, in-box $0.7796, out-of-box
$2.5488 (original $2.5489), and identical violation rates. The harness is
therefore unchanged. The labelled rows (10 models x 4 label budgets x 2 seeds)
were copied from results/baselines/ without refitting, and only the PINN row
is new.

## Files
- `american_put_{full,extrapolate,noisy}.csv`: the v2 PINN row plus the
  copied labelled rows.
- `beat_pinn_counts.csv`: number of labelled models whose seed-mean MAE is
  below the PINN's, per label budget, for v1 and v2.
- `extrapolation_comparison.csv`: out-of-box MAE at 3000 labels.

## Results (v2 PINN: MAE $0.231 on the full grid, in-box $0.209, out-of-box $0.235)

**Full labels.** Number of labelled models (of 10) that beat the PINN:

| Labels | v2 | v1 |
|---|---|---|
| 50 | 2 | 9 |
| 200 | 5 | 9 |
| 1000 | 7 | 10 |
| 3000 | 8 | 10 |

- The two that beat it at 50 labels are bs_iv_net and resnet_silu.
- The two that lose at 3000 labels are poly_ridge ($0.425) and picnn ($0.599).

**Extrapolation (labels only in the box).**
- Out of the box, only bs_iv_net ($0.13) beats the v2 PINN, at every budget.
  The PINN is second. Next come mlp_relu ($2.46) and mlp_tanh ($2.62); kernel
  ridge is at $67.06 and poly ridge at $142.81. (v1: 1-2 of 10 beat the PINN.)
- Inside the box, at 3000 labels, 9 of 10 beat the PINN.
