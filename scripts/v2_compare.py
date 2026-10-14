"""Write results/uq/paper_v2/v1_vs_v2.csv (long format: one row per metric)."""
import pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results/uq"
V = {"v1": R / "paper", "v2": R / "paper_v2"}
rows = []
def add(section, model, method, alpha, n_cal, metric, get):
    d = {"section": section, "model": model, "method": method, "alpha": alpha, "n_cal": n_cal, "metric": metric}
    for k, p in V.items(): d[k] = get(p)
    d["delta_v2_minus_v1"] = d["v2"] - d["v1"]; rows.append(d)
for model in ["n2", "n1", "n0"]:
    D = {k: pd.read_csv(p / "p1p2/p1p2_model_diagnostics_summary.csv").set_index("model").loc[model] for k, p in V.items()}
    S = {k: pd.read_csv(p / "p1p2/p1p2_signed_error_summary.csv").query("model == @model and bucket == 'all'").iloc[0] for k, p in V.items()}
    inv = {v: k for k, v in V.items()}
    for metric, col in [("pool_MAE_usd", "pool_mae_usd_mean"), ("pool_RMSE_usd", "pool_rmse_usd_mean"), ("pool_max_err_usd", "pool_max_usd_mean"),
                        ("spearman_absPDEresid_vs_abserr", "spearman_abs_pde_resid_vs_abs_err_mean"),
                        ("spearman_absLCPresid_vs_abserr", "spearman_abs_lcp_resid_vs_abs_err_mean"), ("train_wall_s", "train_wall_seconds_mean")]:
        if col in D["v1"].index and pd.notna(D["v1"][col]):
            add("point", model, "", "", "", metric, lambda p, c=col: D[inv[p]][c])
    add("point", model, "", "", "", "bias_usd", lambda p: S[inv[p]]["mean_signed_err_usd_mean"])
    add("point", model, "", "", "", "frac_above_ref", lambda p: S[inv[p]]["frac_over_mean"])
    Q = {k: pd.read_csv(p / "p1p2/p1p2_summary.csv").query("model == @model") for k, p in V.items()}
    for _, r in Q["v1"].iterrows():
        for metric in ["coverage_mean", "mean_width_mean", "worst_bucket_cov_mean"]:
            sel = lambda p, r=r, metric=metric: Q[inv[p]].query("n_cal == @r.n_cal and alpha == @r.alpha and method == @r.method")[metric].iloc[0]
            add("conformal", model, r.method, r.alpha, r.n_cal, metric.replace("_mean", ""), sel)
P = {k: pd.read_csv(p / "p4/p4_summary.csv").query("model == 'n2' and tol == 1e-6") for k, p in V.items()}
inv = {v: k for k, v in V.items()}
for _, r in P["v1"].iterrows():
    sel = lambda p, r=r: P[inv[p]].query("method == @r.method and edge == @r.edge and check == @r.check")["rate_mean"].iloc[0]
    add("p4", "n2", f"{r.method}:{r['edge']}", "", "", f"viol_rate_{r.check}", sel)
    if r.check == "strike_monotonicity":
        sel2 = lambda p, r=r: P[inv[p]].query("method == @r.method and edge == @r.edge and check == @r.check")["frac_negative_price_mean"].iloc[0]
        add("p4", "n2", f"{r.method}:{r['edge']}", "", "", "frac_negative_price", sel2)
out = pd.DataFrame(rows); out.to_csv(V["v2"] / "v1_vs_v2.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_rows", 300)
print(out[out.model == "n2"].round(4).to_string(index=False))
