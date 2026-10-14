"""v2 diagnosis step 2: one N2 training with config overrides, scored on the
diagnostic validation grid (paper_v2/diagnosis/diag_grid_crr500.csv, a fixed
53 x 20 grid on m in [0.3, 1.6], tau in [0.05, 1]; NOT a calibration/test pool).

    python scripts/v2_diag_experiment.py --name E2_min --seed 0 --set lcp_form=min
"""
import argparse, json, sys, time
import numpy as np, pandas as pd, torch
from ivuq.pinn import AmericanPINN, AmericanPINNConfig
from ivuq.pinn.black_scholes_pde import pde_residual
sys.path.insert(0, "scripts")
from paper_common import ROOT

OUT = ROOT / "results/uq/paper_v2/diagnosis"
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=6000)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
torch.set_num_threads(1)
over = {}
for kv in a.set:
    k, v = kv.split("=")
    over[k] = {"True": True, "False": False}.get(v, v)
    try: over[k] = float(v) if "." in v or "e" in v else int(v)
    except ValueError: pass
cfg = AmericanPINNConfig(option_type="put", weighting_scheme=over.pop("weighting_scheme", "fixed"), seed=a.seed, epochs=a.epochs, **over)
t0 = time.perf_counter(); pinn = AmericanPINN(cfg).fit(); wall = time.perf_counter() - t0
g = pd.read_csv(OUT / "diag_grid_crr500.csv")
m, tau, ut = g.m.to_numpy(), g.tau.to_numpy(), g.u_crr.to_numpy()
M = torch.tensor(m, dtype=torch.float32).reshape(-1, 1); T = torch.tensor(tau, dtype=torch.float32).reshape(-1, 1)
with torch.no_grad(): u = pinn.model(torch.cat([M, T], 1)).numpy().ravel()
L = pde_residual(pinn.model, M, T, cfg.r, cfg.q, cfg.sigma).detach().numpy().ravel()
h = np.maximum(1 - m, 0); ex = ut <= h + 1e-6; ev = (m >= 0.6) & (m <= 1.4)
e = 100 * (u - ut)
lcp = np.minimum(L, u - h)
res = dict(name=a.name, seed=a.seed, epochs=a.epochs, overrides=" ".join(a.set) or "(v1 config)", wall_s=round(wall, 1),
           mae_eval=np.abs(e[ev]).mean(), bias_eval=e[ev].mean(), frac_over_eval=(e[ev] > 0).mean(), max_abs_eval=np.abs(e[ev]).max(),
           bias_exercise=e[ex & ev].mean(), bias_cont=e[~ex & ev].mean(),
           L_cont_mean=L[~ex].mean(), lcp_res_mean_abs=np.abs(lcp).mean(), frac_u_below_h=(u < h - 1e-4).mean(),
           **{f"final_{k}": v[-1] for k, v in pinn.history.items()})
pd.DataFrame([res]).to_csv(OUT / f"exp_{a.name}_seed{a.seed}.csv", index=False)
print(json.dumps(res, default=float))
