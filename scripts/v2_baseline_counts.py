"""Count labelled models that beat the PINN (mean MAE over seeds) per label budget,
for v1 (results/baselines) and v2 (results/uq/paper_v2/baselines). Writes
results/uq/paper_v2/baselines/beat_pinn_counts.csv and extrapolation_comparison.csv."""
import pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
SRC = {"v1": R / "baselines", "v2": R / "uq/paper_v2/baselines"}
rows, ext = [], []
for v, d in SRC.items():
    for g, col in [("full", "mae_usd"), ("extrapolate", "mae_outside_usd"), ("extrapolate", "mae_inside_usd")]:
        df = pd.read_csv(d / f"american_put_{g}.csv")
        p = df[df.family == "pinn"][col].iloc[0]
        lab = df[df.family != "pinn"].groupby(["n_train", "model"])[col].mean().reset_index()
        for n, gg in lab.groupby("n_train"):
            rows.append(dict(version=v, regime=g, metric=col, n_train=n, pinn_mae_usd=p, n_models=len(gg),
                             n_beat_pinn=int((gg[col] < p).sum()), beaters=";".join(gg[gg[col] < p].model),
                             best_model=gg.loc[gg[col].idxmin(), "model"], best_mae_usd=gg[col].min()))
        if g == "extrapolate" and col == "mae_outside_usd":
            e = lab[lab.n_train == 3000].assign(version=v, pinn_out_usd=p)
            ext.append(e)
out = pd.DataFrame(rows); out.to_csv(SRC["v2"] / "beat_pinn_counts.csv", index=False)
pd.concat(ext).to_csv(SRC["v2"] / "extrapolation_comparison.csv", index=False)
pd.set_option("display.width", 250)
print(out.drop(columns="beaters").round(3).to_string(index=False))
print(pd.concat(ext).pivot(index="model", columns="version", values="mae_outside_usd").round(3))
