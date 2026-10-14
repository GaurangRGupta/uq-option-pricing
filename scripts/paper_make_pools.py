"""Labelled calibration/test pools for P1/P2 (never used for training).

For each seed s: 4000 i.i.d. points m ~ U[0.6,1.4], tau ~ U[0.05,1.0] drawn with
numpy default_rng(100000 + s), labelled with Black-Scholes (European put) and
the 500-step CRR tree (American put). Writes results/uq/paper/pools/pool_seed<s>.csv.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from ivuq.baselines.data import MarketParams, reference_u

from paper_common import PAPER

N_POOL = 4000
M_RANGE = (0.6, 1.4)
TAU_RANGE = (0.05, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = PAPER / "pools"
    out.mkdir(parents=True, exist_ok=True)
    p = MarketParams()
    for s in args.seeds:
        t0 = time.perf_counter()
        rng = np.random.default_rng(100000 + s)
        m = rng.uniform(*M_RANGE, N_POOL)
        tau = rng.uniform(*TAU_RANGE, N_POOL)
        u_eu = reference_u("european", "put", m, tau, p)
        u_am = reference_u("american", "put", m, tau, p, tree_steps=500, max_workers=args.workers)
        pd.DataFrame({"m": m, "tau": tau, "u_eu_bs": u_eu, "u_am_crr500": u_am}).to_csv(out / f"pool_seed{s}.csv", index=False)
        print(f"pool seed {s} in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
