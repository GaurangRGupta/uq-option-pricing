# Code changes for v2 (N2 bias fix)

No git commit was made. The files were edited in place, and the pre-change
copy of each one is kept next to it as `<file>.orig`. The unified diff is
below. The original line endings are kept: the src/ files are CRLF, and CRs
are stripped in this diff for readability.

Every change is opt-in. With default arguments, the code behaves exactly as
in v1:
- `AmericanPINNConfig.lcp_form` defaults to `"product"`.
- `AmericanPINNConfig.american_low_boundary` defaults to `False`.
- `free_boundary_residuals(..., lcp_form="product")`.
- `paper_common.PAPER` is still `results/uq/paper` unless `IVUQ_PAPER_DIR` is set.
- `paper_train_pinns.py` without `--n2-fix` builds the v1 N2.

The European code (`train.py`, `european.py`, `black_scholes_pde.py`) is
untouched, so N0 and N1 were not retrained. Their v1 checkpoints were copied
to paper_v2/models. Re-evaluating them reproduces the v1 numbers to within
3e-5 (`v1_vs_v2.csv`).

New files (no existing file replaced):
- `scripts/v2_diag_v1models.py` (diagnosis step 1)
- `scripts/v2_diag_experiment.py` (diagnosis step 2)
- `scripts/v2_compare.py` (writes v1_vs_v2.csv)
- `tests/test_pinn_american_lcp_form.py` (3 tests: defaults are v1, invalid
  value is rejected, min form = min(L[u], u-h))

How v2 was produced (from repo root, venv `.venv`):

    export IVUQ_PAPER_DIR=$PWD/results/uq/paper_v2
    python scripts/paper_train_pinns.py --model n2 --seed S --threads 1 \
        --n2-fix '{"lcp_form": "min", "american_low_boundary": true, "lambda_complementarity": 10.0}'   # S = 0..9
    cd scripts && python paper_p1p2_conformal.py && python paper_signed_error.py \
        && python paper_p4_arbitrage.py && python paper_figures.py
    cd .. && python scripts/v2_compare.py

```diff
--- a/src/ivuq/pinn/config.py
+++ b/src/ivuq/pinn/config.py
@@ -196,6 +196,21 @@
     curriculum_ramp_epochs: int = 1500
     curriculum_start_frac: float = 0.1
 
+    # v2 (results/uq/paper_v2/DIAGNOSIS.md). Both default to the v1 behaviour,
+    # so the v1 models and every existing test are unchanged.
+    # lcp_form: how the complementarity condition L[u](u-h)=0 is penalised.
+    #   "product" (v1): complementarity_violation = L[u] * (u - h). Weak where
+    #       u - h is small, so a supersolution with L[u] > 0 in the continuation
+    #       region costs almost nothing (the v1 N2 upward bias).
+    #   "min": complementarity_violation = min(L[u], u - h), the standard
+    #       min-form of the LCP. It is zero iff the LCP holds and penalises
+    #       L[u] > 0 at full strength wherever u is above the obstacle.
+    lcp_form: str = "product"
+    # american_low_boundary: put value imposed at m = 0. False (v1) uses the
+    # European value exp(-r tau); True uses the American value 1 (immediate
+    # exercise of a put at S = 0 pays K), consistent with u >= h.
+    american_low_boundary: bool = False
+
     lr: float = 2e-3
     lr_decay_every: int = 2000  # halve lr every this many epochs, 0 disables
     epochs: int = 6000
@@ -204,6 +219,8 @@
     def __post_init__(self) -> None:
         if self.option_type not in ("call", "put"):
             raise ValueError(f"option_type must be 'call' or 'put', got {self.option_type!r}")
+        if self.lcp_form not in ("product", "min"):
+            raise ValueError(f"lcp_form must be 'product' or 'min', got {self.lcp_form!r}")
         if self.weighting_scheme not in ("fixed", "curriculum", "self_adaptive"):
             raise ValueError(
                 "weighting_scheme must be 'fixed', 'curriculum', or 'self_adaptive', "
--- a/src/ivuq/pinn/free_boundary.py
+++ b/src/ivuq/pinn/free_boundary.py
@@ -62,6 +62,7 @@
     q: float,
     sigma: float,
     option_type: str,
+    lcp_form: str = "product",
 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
     """The three (unreduced, per-point) terms of the LCP penalty (3) above.
 
@@ -76,6 +77,10 @@
 
     ineq_violation = torch.relu(-l_u)
     obstacle_violation = torch.relu(h - u)
-    complementarity_violation = l_u * (u - h)
+    if lcp_form == "min":
+        # min-form LCP residual: zero iff L[u] >= 0, u >= h and one of them binds.
+        complementarity_violation = torch.minimum(l_u, u - h)
+    else:
+        complementarity_violation = l_u * (u - h)
 
     return ineq_violation, obstacle_violation, complementarity_violation
--- a/src/ivuq/pinn/train_american.py
+++ b/src/ivuq/pinn/train_american.py
@@ -63,6 +63,8 @@
     u_low_pred = model(torch.cat([m_low, tau_bnd], dim=1))
     u_high_pred = model(torch.cat([m_high, tau_bnd], dim=1))
     u_low_true = boundary_low(tau_bnd, config.r, config.option_type)
+    if config.american_low_boundary and config.option_type == "put":
+        u_low_true = torch.ones_like(tau_bnd)  # American put at S=0: exercise now, worth K
     u_high_true = boundary_high(tau_bnd, config.m_max, config.r, config.q, config.option_type)
     loss_boundary = torch.mean((u_low_pred - u_low_true) ** 2) + torch.mean((u_high_pred - u_high_true) ** 2)
     return loss_terminal, loss_boundary
@@ -118,7 +120,8 @@
             m_int, tau_int = sample_interior(config.n_interior, config.m_max, config.tau_max)
 
         ineq_v, obstacle_v, complementarity_v = free_boundary_residuals(
-            model, m_int, tau_int, config.r, config.q, config.sigma, config.option_type
+            model, m_int, tau_int, config.r, config.q, config.sigma, config.option_type,
+            lcp_form=config.lcp_form,
         )
 
         if config.weighting_scheme == "self_adaptive":
--- a/scripts/paper_common.py
+++ b/scripts/paper_common.py
@@ -2,6 +2,7 @@
 
 from __future__ import annotations
 
+import os
 from pathlib import Path
 
 import numpy as np
@@ -9,7 +10,8 @@
 from ivuq.uq import empirical_coverage, mean_width, moneyness_tenor_buckets
 
 ROOT = Path(__file__).resolve().parents[1]
-PAPER = ROOT / "results" / "uq" / "paper"
+# v1 outputs by default; set IVUQ_PAPER_DIR (e.g. results/uq/paper_v2) to read/write another run.
+PAPER = Path(os.environ["IVUQ_PAPER_DIR"]).resolve() if os.environ.get("IVUQ_PAPER_DIR") else ROOT / "results" / "uq" / "paper"
 BUCKET_NAMES = {
     0: "m<0.9,tau<0.4",
     1: "m<0.9,tau>=0.4",
--- a/scripts/paper_train_pinns.py
+++ b/scripts/paper_train_pinns.py
@@ -21,18 +21,23 @@
 
 from ivuq.pinn import AmericanPINN, AmericanPINNConfig, EuropeanPINN, EuropeanPINNConfig
 
-ROOT = Path(__file__).resolve().parents[1]
-OUT = ROOT / "results" / "uq" / "paper" / "models"
+from paper_common import PAPER
+
+OUT = PAPER / "models"  # v1: results/uq/paper/models; IVUQ_PAPER_DIR overrides
+
+# v2 fix for N2 (results/uq/paper_v2/DIAGNOSIS.md); empty = v1 config.
+N2_FIX: dict = {}
 
 
 def build(model: str, seed: int, epochs: int | None):
     extra = {} if epochs is None else {"epochs": epochs}
+    n2_extra = dict(N2_FIX)
     if model == "n1":
         return EuropeanPINN(EuropeanPINNConfig(option_type="put", lambda_pde=1.0, seed=seed, **extra))
     if model == "n0":
         return EuropeanPINN(EuropeanPINNConfig(option_type="put", lambda_pde=0.0, seed=seed, **extra))
     if model == "n2":
-        return AmericanPINN(AmericanPINNConfig(option_type="put", weighting_scheme="fixed", seed=seed, **extra))
+        return AmericanPINN(AmericanPINNConfig(option_type="put", weighting_scheme="fixed", seed=seed, **extra, **n2_extra))
     raise ValueError(model)
 
 
@@ -42,7 +47,10 @@
     ap.add_argument("--seed", type=int, required=True)
     ap.add_argument("--threads", type=int, default=1)
     ap.add_argument("--epochs", type=int, default=None, help="override only for smoke tests")
+    ap.add_argument("--n2-fix", default="", help='JSON dict of AmericanPINNConfig overrides for N2 (v2), e.g. \'{"lcp_form": "min"}\'')
     args = ap.parse_args()
+    if args.n2_fix:
+        N2_FIX.update(json.loads(args.n2_fix))
     torch.set_num_threads(args.threads)
     OUT.mkdir(parents=True, exist_ok=True)
     tag = f"{args.model}_seed{args.seed}"
```
