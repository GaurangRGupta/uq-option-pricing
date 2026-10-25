"""Train the parameter-conditioned Heston American-put PINN for one underlying and
evaluate against fresh FD solves.

    PYTHONPATH=src python scripts/train_heston_cond.py --symbol SPY --epochs 6000 --w-data 0.0 --tag phys
    PYTHONPATH=src python scripts/train_heston_cond.py --symbol SPY --epochs 6000 --w-data 5.0 --tag anch

--w-data 0 is the pure-physics PINN. Above 0 adds the FD-label anchor term
(reported separately as the anchored variant). Outputs models/heston_cond/<symbol>_<tag>.pt
and results/heston_trading/cond_eval_<symbol>_<tag>.json.
"""
import argparse
import json
from pathlib import Path

import torch

from ivuq.pinn.heston_boxes import box_for
from ivuq.pinn.heston_cond import HestonCondConfig, HestonCondPINN
from ivuq.pinn.heston_cond_coupled import CoupledConfig, HestonCoupledPINN
from ivuq.pinn.heston_cond_data import evaluate, fd_labels, fd_labels_coupled



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--w-data", type=float, default=0.0)
    ap.add_argument("--tag", default="phys")
    ap.add_argument("--n-label-params", type=int, default=400)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-variate", action="store_true")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--n-interior", type=int, default=4000)
    ap.add_argument("--n-data", type=int, default=2000)
    ap.add_argument("--coupled", action="store_true", help="solution + boundary network model")
    ap.add_argument("--w-bdata", type=float, default=0.0)
    ap.add_argument("--r-lo", type=float, default=0.034)
    ap.add_argument("--r-hi", type=float, default=0.043)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    box = box_for(args.symbol, r=(args.r_lo, args.r_hi))
    if args.coupled:
        cfg = CoupledConfig(box=box, epochs=args.epochs, w_data=args.w_data, w_bdata=args.w_bdata, seed=args.seed,
                            hidden_width=args.width, n_interior=args.n_interior, n_data=args.n_data)
        pinn = HestonCoupledPINN(cfg)
        data = None
        if args.w_data > 0 or args.w_bdata > 0:
            print("building FD labels (coupled)", flush=True)
            data = fd_labels_coupled(box, args.n_label_params, 400, 120, seed=args.seed + 1, workers=args.threads)
    else:
        cfg = HestonCondConfig(box=box, epochs=args.epochs, w_data=args.w_data, seed=args.seed,
                               control_variate=args.control_variate, hidden_width=args.width,
                               n_interior=args.n_interior, n_data=args.n_data)
        pinn = HestonCondPINN(cfg)
        data = None
        if args.w_data > 0:
            print("building FD labels", flush=True)
            data = fd_labels(box, args.n_label_params, 400, seed=args.seed + 1, workers=args.threads)
    out = Path("models/heston_cond"); out.mkdir(parents=True, exist_ok=True)
    pinn.fit(data, ckpt=out / f"{args.symbol}_{args.tag}.pt")
    res = evaluate(pinn, workers=args.threads)
    res.update(symbol=args.symbol, tag=args.tag, epochs=args.epochs, w_data=args.w_data, control_variate=args.control_variate, coupled=args.coupled)
    print(json.dumps(res, indent=1))
    Path("results/heston_trading").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(f"results/heston_trading/cond_eval_{args.symbol}_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
