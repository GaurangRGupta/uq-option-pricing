"""v2 diagnosis step 1: inspect the v1 N2 models (no retraining).

For seeds 0-9 of results/uq/paper/models/n2_*: signed error vs CRR-500 on a
diagnostic grid that is NOT a calibration/test pool, split by region, plus
the value at the m=0 and m=m_max edges, the obstacle u>=h, and the sign of
L[u] in the continuation region. Writes paper_v2/diagnosis/v1_model_diag.csv.
"""
import sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "scripts")
from paper_models import load_pinn
from ivuq.baselines.data import MarketParams, reference_u
from ivuq.pinn.black_scholes_pde import pde_residual
from paper_common import ROOT

OUT = ROOT / "results/uq/paper_v2/diagnosis"
torch.set_num_threads(8)
p = MarketParams()
# diagnostic grid: m 0.30..1.60 (41), tau 0.05..1.0 (20)
mg, tg = np.meshgrid(np.linspace(0.3, 1.6, 53), np.linspace(0.05, 1.0, 20))
m, tau = mg.ravel(), tg.ravel()
cache = OUT / "diag_grid_crr500.csv"
if cache.exists():
    g = pd.read_csv(cache)
else:
    g = pd.DataFrame({"m": m, "tau": tau, "u_crr": reference_u("american", "put", m, tau, p, tree_steps=500, max_workers=8)})
    g.to_csv(cache, index=False)
m, tau, ut = g.m.to_numpy(), g.tau.to_numpy(), g.u_crr.to_numpy()
h = np.maximum(1 - m, 0)
ex = ut <= h + 1e-6           # true exercise region
rows = []
for s in range(10):
    pinn = load_pinn("n2", s); net = pinn.model
    M = torch.tensor(m, dtype=torch.float32).reshape(-1, 1); T = torch.tensor(tau, dtype=torch.float32).reshape(-1, 1)
    with torch.no_grad():
        u = net(torch.cat([M, T], 1)).numpy().ravel()
    L = pde_residual(net, M, T, p.r, p.q, p.sigma).detach().numpy().ravel()
    e = 100 * (u - ut)
    tt = torch.linspace(0, 1, 101).reshape(-1, 1)
    with torch.no_grad():
        u0 = net(torch.cat([torch.zeros_like(tt), tt], 1)).numpy().ravel()
        um = net(torch.cat([torch.full_like(tt, 3.0), tt], 1)).numpy().ravel()
        mt = torch.linspace(0, 3, 301).reshape(-1, 1)
        uT = net(torch.cat([mt, torch.zeros_like(mt)], 1)).numpy().ravel()
    hT = np.maximum(1 - mt.numpy().ravel(), 0)
    cont = ~ex
    rows.append(dict(seed=s,
        bias_all=e.mean(), frac_over=(e > 0).mean(),
        bias_exercise=e[ex].mean(), bias_cont_itm=e[cont & (m < 1)].mean(), bias_otm=e[m >= 1].mean(),
        bias_otm_m_ge_1p3=e[m >= 1.3].mean(),
        u_at_m0_mean=u0.mean(), u_at_m0_tau1=u0[-1], amer_value_m0=1.0, euro_value_m0_tau1=float(np.exp(-p.r)),
        u_at_mmax_maxabs=np.abs(um).max(),
        terminal_maxabs_err=np.abs(uT - hT).max(), terminal_mean_err_usd=100 * np.mean(uT - hT),
        frac_u_below_h=(u < h - 1e-4).mean(),
        L_cont_mean=L[cont].mean(), L_cont_median=np.median(L[cont]), L_cont_frac_pos=(L[cont] > 0).mean(),
        L_cont_otm_mean=L[cont & (m >= 1.2)].mean()))
df = pd.DataFrame(rows)
df.to_csv(OUT / "v1_model_diag.csv", index=False)
pd.set_option("display.width", 250)
print(df.round(4).T.to_string())
print("\nmean over seeds:\n", df.mean().round(4).to_string())
