"""Build every table of the paper from CSVs under ../results/.

    python make_tables.py      (run from paper_ieee/, with the project venv)

For each table this writes tables/<name>.csv (the numbers shown, with the
source file named in a 'source' column or in tables/SOURCES.md) and
tables/<name>.tex (the LaTeX body that main.tex inputs). No number in the
paper tables is typed by hand.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = __import__("pathlib").Path("/workspace/ivuq/paper_ieee")
RES = HERE.parent / "results"
PAP = RES / "uq" / "paper_v2"
TAB = __import__("pathlib").Path("/tmp/v2tab")
TAB.mkdir(exist_ok=True)
SOURCES: list[str] = []
VALUES: dict[str, str] = {}

NAMES = {
    "poly_ridge": "Polynomial ridge (deg. 6)",
    "kernel_ridge": "Kernel ridge (RBF, CV)",
    "random_forest": r"Random forest \cite{ho1995random,breiman2001random}",
    "grad_boosting": r"Gradient boosting \cite{friedman2001greedy}",
    "mlp_tanh": "MLP, tanh, 4x64",
    "mlp_relu": "MLP, ReLU, 4x64",
    "resnet_silu": "Residual MLP, SiLU",
    "fourier_mlp": "Fourier-feature MLP",
    "picnn": r"Partial ICNN \cite{amos2017input}",
    "bs_iv_net": "BS-embedded IV net",
    "pinn_n2": "PINN N2 (no labels)",
}
METH = {"vanilla": "Split", "normalised_pde": "Norm. $|L[u]|$", "normalised_lcp": "Norm. LCP",
        "mondrian": "Mondrian", "normalised": "Normalised"}


def f(x, d=3):
    return "n/a" if pd.isna(x) else f"{x:.{d}f}"


def write(name, df, tex, srcs):
    df.to_csv(TAB / f"{name}.csv", index=False)
    (TAB / f"{name}.tex").write_text(tex)
    SOURCES.append(f"- `tables/{name}.csv` / `.tex`: " + "; ".join(srcs))


def t_baselines():
    full = pd.read_csv(RES / "baselines" / "american_put_full.csv")
    ext = pd.read_csv(RES / "baselines" / "american_put_extrapolate.csv")
    rows = []
    for key, name in NAMES.items():
        if key == "pinn_n2":
            a = full[full.model == key]
            e = ext[ext.model == key]
            r = dict(model=name, key=key, mae_n50=a.mae_usd.mean(), mae_n3000=a.mae_usd.mean(),
                     out_mae_n3000=e.mae_outside_usd.mean(), in_mae_n3000=e.mae_inside_usd.mean(),
                     viol_convex=a.viol_convex.mean(), viol_calendar=a.viol_calendar.mean(), n_seeds=a.seed.nunique())
        else:
            a50 = full[(full.model == key) & (full.n_train == 50)]
            a = full[(full.model == key) & (full.n_train == 3000)]
            e = ext[(ext.model == key) & (ext.n_train == 3000)]
            r = dict(model=name, key=key, mae_n50=a50.mae_usd.mean(), mae_n3000=a.mae_usd.mean(),
                     out_mae_n3000=e.mae_outside_usd.mean(), in_mae_n3000=e.mae_inside_usd.mean(),
                     viol_convex=a.viol_convex.mean(), viol_calendar=a.viol_calendar.mean(), n_seeds=a.seed.nunique())
        rows.append(r)
    df = pd.DataFrame(rows)
    lines = []
    for _, r in df.iterrows():
        lines.append(f"{r.model} & {f(r.mae_n50)} & {f(r.mae_n3000)} & {f(r.in_mae_n3000)} & {f(r.out_mae_n3000, 2)} & {f(r.viol_convex, 2)} & {f(r.viol_calendar, 2)} \\\\")
        if r.key == "bs_iv_net":
            lines.append("\\midrule")
    write("table1_baselines", df, "\n".join(lines) + "\n",
          ["results/baselines/american_put_full.csv (n_train 50 and 3000, mean of seeds 0-1; PINN row n_train 0)",
           "results/baselines/american_put_extrapolate.csv (n_train 3000; mae_inside_usd, mae_outside_usd)"])
    return df


def t_point():
    d = pd.read_csv(PAP / "p1p2" / "p1p2_model_diagnostics_summary.csv")
    s = pd.read_csv(PAP / "p1p2" / "p1p2_signed_error_summary.csv")
    s = s[s.bucket == "all"]
    df = d.merge(s, on="model")
    order = ["n0", "n1", "n2"]
    df = df.set_index("model").loc[order].reset_index()
    keep = ["model", "pool_mae_usd_mean", "pool_mae_usd_std", "pool_rmse_usd_mean", "pool_max_usd_mean",
            "mean_signed_err_usd_mean", "frac_over_mean", "spearman_abs_pde_resid_vs_abs_err_mean",
            "spearman_abs_pde_resid_vs_abs_err_std", "spearman_abs_lcp_resid_vs_abs_err_mean",
            "spearman_abs_lcp_resid_vs_abs_err_std", "train_wall_seconds_mean"]
    df = df[keep]
    lab = {"n0": "N0 (Eur., no PDE)", "n1": "N1 (Eur., PDE)", "n2": "N2 (Amer., LCP)"}
    lines = []
    for _, r in df.iterrows():
        lcp = "n/a" if pd.isna(r.spearman_abs_lcp_resid_vs_abs_err_mean) else f(r.spearman_abs_lcp_resid_vs_abs_err_mean, 2)
        lines.append(f"{lab[r.model]} & {f(r.pool_mae_usd_mean, 2)} & {f(r.pool_max_usd_mean, 2)} & {r.mean_signed_err_usd_mean:+.2f} & "
                     f"{f(r.frac_over_mean, 3)} & {f(r.spearman_abs_pde_resid_vs_abs_err_mean, 2)} & {lcp} & {r.train_wall_seconds_mean:.0f} \\\\")
    write("table2_point", df, "\n".join(lines) + "\n",
          ["results/uq/paper/p1p2/p1p2_model_diagnostics_summary.csv", "results/uq/paper/p1p2/p1p2_signed_error_summary.csv (bucket = all)"])
    return df


def t_conformal():
    s = pd.read_csv(PAP / "p1p2" / "p1p2_summary.csv")
    s = s[s.n_cal == 800]
    order = [("n0", "vanilla"), ("n0", "normalised_pde"), ("n0", "mondrian"),
             ("n1", "vanilla"), ("n1", "normalised_pde"), ("n1", "mondrian"),
             ("n2", "vanilla"), ("n2", "normalised_pde"), ("n2", "normalised_lcp"), ("n2", "mondrian")]
    rows, lines = [], []
    lab = {"n0": "N0", "n1": "N1", "n2": "N2"}
    for mod, meth in order:
        r10 = s[(s.model == mod) & (s.method == meth) & np.isclose(s.alpha, 0.10)].iloc[0]
        r05 = s[(s.model == mod) & (s.method == meth) & np.isclose(s.alpha, 0.05)].iloc[0]
        rows.append(dict(model=mod, method=meth,
                         cov10=r10.coverage_mean, cov10_std=r10.coverage_std, width10=r10.mean_width_mean, worst10=r10.worst_bucket_cov_mean,
                         cov05=r05.coverage_mean, cov05_std=r05.coverage_std, width05=r05.mean_width_mean, worst05=r05.worst_bucket_cov_mean))
        if meth == "vanilla" and mod != "n0":
            lines.append("\\midrule")
        lines.append(f"{lab[mod]} & {METH[meth]} & {f(r10.coverage_mean)} & {f(r10.mean_width_mean, 2)} & {f(r10.worst_bucket_cov_mean, 2)} & "
                     f"{f(r05.coverage_mean)} & {f(r05.mean_width_mean, 2)} & {f(r05.worst_bucket_cov_mean, 2)} \\\\")
    df = pd.DataFrame(rows)
    maxstd = float(df[["cov10_std", "cov05_std"]].max().max())
    VALUES["maxcovstd"] = f"{maxstd:.3f}"
    write("table3_conformal", df, "\n".join(lines) + "\n", ["results/uq/paper/p1p2/p1p2_summary.csv (n_cal = 800)"])
    return df


def t_p3():
    s = pd.read_csv(PAP / "p3" / "p3_summary.csv")
    rows, lines = [], []
    for a in (0.10, 0.05):
        for meth in ("vanilla", "normalised", "mondrian"):
            r = s[(s.experiment == "heteroscedastic") & (s.method == meth) & np.isclose(s.alpha, a)].iloc[0]
            rows.append(dict(alpha=a, method=meth, coverage=r.coverage_mean, coverage_std=r.coverage_std, width=r.mean_width_mean,
                             cov_near=r.coverage_near_atm_mean, cov_far=r.coverage_away_atm_mean, worst=r.worst_bucket_cov_mean, kappa=r.kappa_mean))
            lines.append(f"{a:.2f} & {METH[meth]} & {f(r.coverage_mean)} & {f(r.mean_width_mean)} & {f(r.coverage_near_atm_mean)} & {f(r.coverage_away_atm_mean)} \\\\")
        if a == 0.10:
            lines.append("\\midrule")
    df = pd.DataFrame(rows)
    homo = s[s.experiment == "homoscedastic"][["alpha", "coverage_mean", "coverage_std"]]
    homo.to_csv(TAB / "table4_p3_homoscedastic.csv", index=False)
    kap = df[df.method == "normalised"].kappa.iloc[0]
    VALUES["kappafit"] = f"{kap:.1f}"
    write("table4_p3", df, "\n".join(lines) + "\n", ["results/uq/paper/p3/p3_summary.csv (experiment heteroscedastic; homoscedastic rows in table4_p3_homoscedastic.csv)"])
    return df


def t_p4():
    s = pd.read_csv(PAP / "p4" / "p4_summary.csv")
    s = s[np.isclose(s.tol, 1e-6)]
    series = [("reference_american", "reference", "point", "CRR reference (Amer.)"),
              ("reference_european", "reference", "point", "BS reference (Eur.)"),
              ("n1", "pinn", "point", "N1 point price"),
              ("n2", "pinn", "point", "N2 point price"),
              ("n2", "vanilla", "lo", "N2 split, lower edge"),
              ("n2", "vanilla", "hi", "N2 split, upper edge"),
              ("n2", "normalised_lcp", "lo", "N2 norm. LCP, lower"),
              ("n2", "normalised_lcp", "hi", "N2 norm. LCP, upper"),
              ("n2", "mondrian", "lo", "N2 Mondrian, lower"),
              ("n2", "mondrian", "hi", "N2 Mondrian, upper")]
    rows, lines = [], []
    for mod, meth, edge, lab in series:
        sub = s[(s.model == mod) & (s.method == meth) & (s.edge == edge)].set_index("check")
        r = dict(series=lab, model=mod, method=meth, edge=edge,
                 monotone=sub.loc["strike_monotonicity", "rate_mean"], convex=sub.loc["strike_convexity", "rate_mean"],
                 calendar=sub.loc["calendar_spread", "rate_mean"], neg=sub["frac_negative_price_mean"].iloc[0])
        rows.append(r)
        lines.append(f"{lab} & {f(r['monotone'])} & {f(r['convex'])} & {f(r['calendar'])} & {f(r['neg'], 2)} \\\\")
    df = pd.DataFrame(rows)
    write("table5_p4", df, "\n".join(lines) + "\n", ["results/uq/paper/p4/p4_summary.csv (tol = 1e-6, mean over 10 seeds)"])
    return df


if __name__ == "__main__":
    for fn in (t_baselines, t_point, t_conformal, t_p3, t_p4):
        print(fn().round(4).to_string(index=False))
    (TAB / "values.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in VALUES.items()))
    SOURCES.append("- `tables/values.tex`: maxcovstd = max seed std of coverage in table3 (p1p2_summary.csv); kappafit = mean fitted kappa, P3 normalised (p3_summary.csv)")
    (TAB / "SOURCES.md").write_text("# Source of each table\n\nGenerated by `paper_ieee/make_tables.py` from the files below. Nothing typed by hand.\n\n" + "\n".join(SOURCES) + "\n")
