# v2 update plan: what would change in main.tex if the team adopts the N2 fix

Status: **main.tex is NOT edited.** This file lists every spot that depends
on N2 numbers or on the v1 N2 training, so the team can decide while
hand-rewriting the text.
- Sources: `results/uq/paper_v2/` (RESULTS.md, v1_vs_v2.csv, DIAGNOSIS.md).
- Rebuilt tables: `results/uq/paper_v2/tables_preview/*.tex`. These are
  produced by the same make_tables.py, pointed at paper_v2, and can be
  dropped into `tables/` if v2 is adopted.
- Line numbers refer to the current main.tex (v2 similarity edit) and will
  drift as you rewrite.

## 0. Decision to make first
v2 meets its pre-registered rule: bias +$3.13 down to +$0.22, lower in all
10 seeds. Three findings change, not just numbers:
1. **Bands:** the N2 band is about 11x narrower ($10.05 to $0.93). Coverage
   is unchanged.
2. **LCP-normalised band:** it **no longer beats split.** It gives $1.13 vs
   $0.93, and its rank correlation drops from 0.77 to 0.38. It still clearly
   beats the raw residual (0.38 vs 0.01; $1.13 vs $1.97).
3. **Calendar arbitrage:** N2 now breaks it on 16.8% of pairs (v1: 0%).
   Convexity violations halve (26% to 12%).

So contribution 2 ("the residual must match the complementarity structure")
survives only in its weaker form: the LCP residual beats the raw residual.
The claim "LCP normalisation narrows the band" must go. The title's
"The Difficulty Score Must Respect Complementarity" is still defensible
against the raw residual. Consider softening it, for example to "...: Why
the Raw PDE Residual Fails for American Options".
- **Option A (adopt v2):** apply sections 1 to 6 below.
- **Option B (keep v1):** keep all numbers and change only the "N2 bias"
  paragraph (section 5). It can now name the tested cause instead of "we
  have not tested these causes", and point to v2 as future work.

## 1. Tables (replace the .tex files; only N2 rows change)

**Table II** (`tables/table2_point.tex`), N2 row:

| Column | v1 | v2 |
|---|---|---|
| MAE | 3.13 | 0.29 |
| Max | 6.89 | 0.84 |
| Bias | +3.13 | +0.22 |
| Above | 0.996 | 0.837 |
| rho, abs(L) | -0.01 | 0.01 |
| rho, LCP | 0.77 | 0.38 |
| Train s | 229 | 165 |

**Table III** (`tables/table3_conformal.tex`), N2 rows, n_cal = 800.
Each cell is v1 -> v2, in the order Cov / Width / Worst.

| Score | alpha = 0.10 | alpha = 0.05 |
|---|---|---|
| Split | 0.900 / 10.05 / 0.71 -> 0.901 / 0.93 / 0.66 | 0.950 / 11.08 / 0.82 -> 0.951 / 1.04 / 0.78 |
| Norm. abs(L[u]) | 0.900 / 11.89 / 0.76 -> 0.901 / 1.97 / 0.65 | 0.950 / 12.79 / 0.85 -> 0.950 / 2.24 / 0.81 |
| Norm. LCP | 0.900 / 9.01 / 0.71 -> 0.900 / 1.13 / 0.64 | 0.950 / 9.96 / 0.81 -> 0.951 / 1.30 / 0.80 |
| Mondrian | 0.904 / 8.67 / 0.86 -> 0.904 / 0.85 / 0.86 | 0.954 / 9.15 / 0.92 -> 0.954 / 0.92 / 0.92 |

The N0 and N1 rows are unchanged, and so is the caption's \maxcovstd
(values.tex is identical).

**Table V** (`tables/table5_p4.tex`), N2 rows. Each cell is v1 -> v2, in
the order Monot. / Convex / Calendar / Price<0.

| Series | v1 -> v2 |
|---|---|
| point price | 0.010 / 0.260 / 0.000 / 0.00 -> 0.039 / 0.121 / 0.168 / 0.02 |
| split, lower | 0.059 / 0.260 / 0.000 / 0.36 -> 0.053 / 0.121 / 0.168 / 0.20 |
| split, upper | 0.001 / 0.260 / 0.000 / 0.00 -> 0.033 / 0.121 / 0.168 / 0.00 |
| norm. LCP, lower | 0.131 / 0.388 / 0.042 / 0.35 -> 0.113 / 0.266 / 0.229 / 0.20 |
| norm. LCP, upper | 0.028 / 0.399 / 0.077 / 0.00 -> 0.042 / 0.226 / 0.240 / 0.00 |
| Mondrian, lower | 0.032 / 0.321 / 0.110 / 0.32 -> 0.051 / 0.145 / 0.164 / 0.18 |
| Mondrian, upper | 0.016 / 0.322 / 0.000 / 0.00 -> 0.033 / 0.148 / 0.180 / 0.00 |

**Table I** (baselines): no file change. It comes from the earlier baseline
run (`results/baselines/american_put_full.csv`), whose "PINN N2" row
($2.313 / 2.55 out-of-box) used the v1 loss on a different grid. If v2 is
adopted, choose one:
- (a) Keep the table and add to the caption: "PINN row: earlier run with
  the v1 loss; see Table II for the v2 PINN".
- (b) Rerun the baseline harness with the v2 N2 config. This is not done
  yet and is outside this task.

Note for the text: against the v2 pool MAE of $0.29 (a different grid, so
not like-for-like), only 3 of 10 models with 50 labels and 8 of 10 with
3000 labels would be more accurate. The sentence "with 50 labels, nine of
ten do" depends on the Table I PINN row. It stays true only under option (a).

## 2. Abstract (line 27)

| Old text | v2 replacement fact |
|---|---|
| "including an American PINN that overprices by $3.13 on average: the bias costs width, not coverage" | The American PINN has a residual bias of +$0.22 (MAE $0.29) after the loss fix. Keep "coverage is nominal for every model". The "bias costs width" point now rests on v1 vs v2: the band shrinks from $10.05 to $0.93. Only say this if the team wants v1 in the paper as an ablation. |
| "raw PDE residual is uninformative ... (rank correlation -0.01) and widens the band" | rank correlation 0.01, band $1.97 vs $0.93 split. Still true. |
| "complementarity residual tracks the error (rank correlation 0.77) and narrows the band from $10.05 to $9.01 at equal coverage" | rank correlation 0.38. It is narrower than the raw residual ($1.13 vs $1.97) but **wider than split ($0.93)**. The narrowing claim must be removed. |
| "Mondrian buckets give the best worst-bucket coverage (0.86 against 0.71)" | 0.86 against 0.66 |

Every abstract number must appear in a table: $0.29, $0.22, 0.01, 0.38,
$1.13, $1.97, $0.93, 0.86 and 0.66 are all in the v2 Tables II and III.

## 3. Introduction
- **Contribution 2** (line 41): "and that the complementarity residual ...
  is a good one". Change to "a better one than the raw residual, though
  plain split is narrower once the PINN is unbiased". If adopting v2, add a
  contribution: "a diagnosis and fix of the N2 bias (min-form LCP penalty)".
- **Limits** (line 43): "systematic upward bias of $3.13". Change to
  "a residual upward bias of $0.22 (84% of points above the reference,
  mostly deep out of the money with long tenor, +$0.35)".

## 4. Method and Experiments
- **Line 73 and eq. (4) (`eq:la`):** "three LCP penalties, each with weight
  1" with the product term (L[u](u-h))^2. In v2, the third term is
  10 x mean(min(L[u], u-h)^2), and the American put uses u(0, tau) = 1 at
  m = 0 instead of e^{-r tau}. If the text describes the boundary g for N2,
  change it. Line 66 (L[h] = r in the exercise region) is unchanged.
- **Line 99, timing:** "The 30 trainings took 12 min 24 s ...; 229 s (N2)"
  and "All conformal evaluations took 62 s". In v2, N0 and N1 are the v1
  runs, and the 10 N2 runs took 5 min 20 s (165 s per run). The conformal
  evaluation took 32.8 s. Suggested wording: "N0 and N1: 157 s and 167 s per
  run; N2: 165 s per run (10 runs in 5 min 20 s); conformal evaluation 33 s."
- **Line 132 (point accuracy):** "N2 has an MAE of $3.13 that is almost all
  bias: it prices 99.6% of points above". Change to "$0.29, bias +$0.22,
  83.7% above". "For N2 it carries no signal (rho = -0.01), while the LCP
  residual has rho = 0.77": change to 0.01 and 0.38. "narrow band (median
  0.057)": the v2 seed-0 median abs(L[u]) is 0.0071 (fig_p2_residual_vs_error.csv).
  "The LCP residual removes this part and rises with the error" now holds
  mainly on the exercise-region branch, so soften it.
- **Line 156 (conformal):**
  - "This holds for N2 too, despite its bias ... the bias shows up as
    width": keep only as v1 vs v2 (N2 split width $10.05 to $0.93).
  - "N2 band ($11.89 against $10.05)": change to $1.97 against $0.93.
  - "LCP residual gives $9.01 against $10.05 at the same coverage, but its
    worst bucket (0.71) is no better than split": change to $1.13 against
    $0.93 (wider) with worst bucket 0.64 vs 0.66.
  - "0.86 against 0.71 for split on N2": change to 0.86 against 0.66.
  - "under-covers deep in-the-money long tenors (0.71)": in v2 split's worst
    bucket is deep in-the-money **short** tenors (m < 0.9, tau < 0.4: 0.69).
  - "Mondrian keeps each bucket between 0.902 and 0.910": change to 0.901
    and 0.906.
  - "they are wide because the PINN sits above it": in v2 the bands are
    narrow and close to CRR.
- **Line 189 (P4):**
  - "26% (N2)": change to 12%.
  - "its lower edge raises monotonicity violations from 0.010 to 0.059":
    change to 0.039 to 0.053.
  - "About a third of lower edges are negative": change to about a fifth
    (0.18 to 0.20).
  - **New fact to state:** N2 point prices now break the calendar check in
    17% of pairs (v1: 0) and 2% of point prices are negative.
  - "A constant split band keeps the convexity and calendar rates of the
    point price" still holds (0.121 / 0.168 on both edges).

## 5. Figure captions and Discussion
- **Fig. 1 (line 137):** "The raw residual stays in a narrow band for any
  error; the LCP residual rises with the error". In v2 the LCP residual
  rises with the error mainly in the exercise region (where it equals the
  obstacle gap) and is weak elsewhere (rho 0.38). Swap to the v2 figure
  `results/uq/paper_v2/figures/fig_p2_residual_vs_error.pdf`.
- **Fig. 2 (line 161):** "Only Mondrian is close to target in every bucket".
  Still true in v2 (fig_p2_bucket_coverage_a10).
- **Fig. 3 (line 168):** "the PINN lies above it, so the bands are wide and
  the lower edges fall below zero out of the money". In v2 the PINN is close
  to CRR and the bands are narrow (about $1). Lower edges still dip below
  zero far out of the money.
- **Line 205 (why keep the PINN):** "even a biased PINN gives bands with
  valid marginal coverage. The cost of the bias is width, so a better PINN
  gives narrower bands at once, as N1 against N0 shows". v2 adds a second
  example: N2 v1 vs v2, $10.05 to $0.93.
- **Line 207 (which residual):** "The LCP residual fixes this and is a much
  better error proxy". Change to "a better proxy than abs(L[u]) (0.38 vs
  0.01), but once the bias is fixed, no residual scaling beats split on
  width; Mondrian is best".
- **Line 209 (the N2 bias):** replace in both options. Facts to use:
  - The tested cause is the product-form penalty L[u](u-h), which barely
    penalises L[u] > 0 where u - h is small. The v1 net was a supersolution
    with L[u] of about 0.042 in the continuation region.
  - The m = 0 boundary was tested and is not the cause (alone it changes the
    bias from +$2.50 to +$2.41).
  - The min-form penalty with weight 10 plus the American edge value cuts
    the bias to +$0.22.
  - Remaining issues: calendar violations (17%), a slightly violated
    obstacle u >= h, and +$0.35 in the deep out-of-the-money long-tenor
    bucket.
  - The fix was chosen on a separate validation grid, not on calibration or
    test data.
  - Option B wording: "we later traced the bias to ...; a fixed model is
    reported in the project repository".

## 6. Conclusion (lines 214-216)
- "including a biased American PINN": under option A, change to "including
  an American PINN with a small residual bias".
- "the complementarity residual is a good one": change to "better than the
  raw residual".
- Future work "fix the N2 bias": change to "remove the remaining calendar
  violations and obstacle undershoot of the fixed N2".

## 7. Files the paper would point to
Under option A, every N2 number moves:
- `results/uq/paper/...` becomes `results/uq/paper_v2/...`.
- `tables/SOURCES.md` must name paper_v2.
- The figures `fig_p2_*` should be copied from `results/uq/paper_v2/figures/`.
- Rerun `make_tables.py` pointed at paper_v2. This needs a one-line path
  change, PAP = RES / "uq" / "paper_v2". The preview copy is at
  results/uq/paper_v2/tables_preview/make_tables_v2.py, and its outputs are in tables_preview/.
