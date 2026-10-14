"""Train one reporting-config PINN (N1, N0 or N2 put) for the paper and save it.

    python scripts/paper_train_pinns.py --model n1 --seed 0

Writes results/uq/paper/models/<model>_seed<seed>.pt (state_dict + config) and
<model>_seed<seed>.json (wall time, final losses). Config = dataclass defaults
in src/ivuq/pinn/config.py (the reporting config). See results/uq/paper/PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import time
from pathlib import Path

import torch

from ivuq.pinn import AmericanPINN, AmericanPINNConfig, EuropeanPINN, EuropeanPINNConfig

from paper_common import PAPER

OUT = PAPER / "models"  # v1: results/uq/paper/models; IVUQ_PAPER_DIR overrides

# v2 fix for N2 (results/uq/paper_v2/DIAGNOSIS.md); empty = v1 config.
N2_FIX: dict = {}


def build(model: str, seed: int, epochs: int | None):
    extra = {} if epochs is None else {"epochs": epochs}
    n2_extra = dict(N2_FIX)
    if model == "n1":
        return EuropeanPINN(EuropeanPINNConfig(option_type="put", lambda_pde=1.0, seed=seed, **extra))
    if model == "n0":
        return EuropeanPINN(EuropeanPINNConfig(option_type="put", lambda_pde=0.0, seed=seed, **extra))
    if model == "n2":
        return AmericanPINN(AmericanPINNConfig(option_type="put", weighting_scheme="fixed", seed=seed, **extra, **n2_extra))
    raise ValueError(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["n1", "n0", "n2"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=None, help="override only for smoke tests")
    ap.add_argument("--n2-fix", default="", help='JSON dict of AmericanPINNConfig overrides for N2 (v2), e.g. \'{"lcp_form": "min"}\'')
    args = ap.parse_args()
    if args.n2_fix:
        N2_FIX.update(json.loads(args.n2_fix))
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model}_seed{args.seed}"
    pinn = build(args.model, args.seed, args.epochs)
    t0 = time.perf_counter()
    pinn.fit()
    wall = time.perf_counter() - t0
    torch.save({"state_dict": pinn.model.state_dict(), "config": dataclasses.asdict(pinn.config)}, OUT / f"{tag}.pt")
    meta = {
        "model": args.model,
        "seed": args.seed,
        "threads": args.threads,
        "wall_seconds": wall,
        "config": dataclasses.asdict(pinn.config),
        "final_losses": {k: v[-1] for k, v in pinn.history.items()},
        "torch": torch.__version__,
        "host": platform.node(),
        "pid": os.getpid(),
    }
    (OUT / f"{tag}.json").write_text(json.dumps(meta, indent=2))
    print(f"{tag} done in {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
