"""Obstacle check u >= h for N2 on the P2 pools, v1 and v2 models (10 seeds each).
Writes results/uq/paper_v2/p1p2/p1p2_obstacle_summary.csv (and _raw.csv)."""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "scripts")
import paper_models, paper_common
from pathlib import Path
R = Path(paper_common.ROOT) / "results/uq"
torch.set_num_threads(8)
rows = []
for v, d in [("v1", R / "paper"), ("v2", R / "paper_v2")]:
    paper_models.PAPER = d
    for s in range(10):
        pool = pd.read_csv(d / "pools" / f"pool_seed{s}.csv")
        m, tau = pool.m.to_numpy(), pool.tau.to_numpy()
        u = paper_models.predictors("n2", s)["pde"].predict(m, tau)
        h = np.maximum(1 - m, 0); gap = 100 * (u - h)
        rows.append(dict(version=v, seed=s, frac_below_h_1c=(gap < -0.01).mean(), frac_below_h_any=(gap < 0).mean(),
                         mean_shortfall_usd=np.where(gap < 0, -gap, 0).mean(), max_shortfall_usd=max(0.0, -gap.min())))
df = pd.DataFrame(rows)
out = R / "paper_v2/p1p2"
df.to_csv(out / "p1p2_obstacle_raw.csv", index=False)
summ = df.groupby("version").agg(["mean", "std"]).drop(columns="seed"); summ.columns = [f"{a}_{b}" for a, b in summ.columns]
summ.reset_index().to_csv(out / "p1p2_obstacle_summary.csv", index=False)
print(summ.round(4).T)
