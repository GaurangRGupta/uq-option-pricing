"""Rerun the PINN row of the labelled-baseline harness (Table I) with v2 N2 settings.

Same harness (ivuq.baselines.experiments.run_regime, unchanged), same 25 x 9
test grid, same regimes and scoring as results/baselines/american_put_*.csv.
The PINN is trained exactly as scripts/run_baseline_comparison.py does
(AmericanPINNConfig defaults, weighting "fixed", seed 0), plus the v2 fix.
Labelled rows are not refit; they are copied from results/baselines/ (the
harness, label sampling and models are unchanged), so only the PINN row is new.

    python scripts/v2_baseline_pinn_row.py --variant v2   # writes results/uq/paper_v2/baselines/
    python scripts/v2_baseline_pinn_row.py --variant v1   # reproduction check of the old PINN row
"""
import argparse, json, sys, time
from pathlib import Path
import pandas as pd, torch
from ivuq.baselines import MarketParams, pinn_predictor, run_regime
from ivuq.pinn import AmericanPINN, AmericanPINNConfig

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/uq/paper_v2/baselines"
V2_FIX = {"lcp_form": "min", "american_low_boundary": True, "lambda_complementarity": 10.0}
REGIMES = ["full", "extrapolate", "noisy"]

ap = argparse.ArgumentParser(); ap.add_argument("--variant", choices=["v1", "v2"], required=True)
a = ap.parse_args()
torch.set_num_threads(1)
OUT.mkdir(parents=True, exist_ok=True)
p = MarketParams()
fix = V2_FIX if a.variant == "v2" else {}
t0 = time.perf_counter()
model = AmericanPINN(AmericanPINNConfig(option_type="put", weighting_scheme="fixed", r=p.r, q=p.q, sigma=p.sigma, **fix)).fit()
train_s = time.perf_counter() - t0
ext = {"pinn_n2": (pinn_predictor(model), train_s)}
for g in REGIMES:
    row = run_regime(g, "american", "put", p, n_list=[50], seeds=[], external=ext)
    old = pd.read_csv(ROOT / f"results/baselines/american_put_{g}.csv")
    if a.variant == "v2":
        labelled = old[old.family != "pinn"]
        pd.concat([row, labelled], ignore_index=True).to_csv(OUT / f"american_put_{g}.csv", index=False)
    else:
        row.to_csv(OUT / f"check_v1_pinn_row_{g}.csv", index=False)
print(json.dumps({"variant": a.variant, "train_seconds": train_s}))
