"""Low-cost sampling experiments (Figure 7).

1. ODE error vs NFE: compare fixed-step solutions to high-NFE reference.
2. FID vs NFE: compute FID at various NFE for Euler, Midpoint, RK4.

Usage:
    python evaluate_low_cost.py --ckpt runs/fm_ot_imagenet32/ckpt.pt --mode error --dataset imagenet
    python evaluate_low_cost.py --ckpt runs/fm_ot_imagenet32/ckpt.pt --mode fid --dataset imagenet
"""
from __future__ import annotations
import os
import argparse
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from sample import load_ckpt, to_img
from src.solvers import SOLVERS, midpoint_sample
from evaluate_fid import generate_samples, save_images_to_dir, compute_fid


def ode_error_experiment(net, image_size: int, device: str, out_dir: str,
                         n_seeds: int = 256, ref_nfe: int = 1000,
                         nfe_list: list[int] | None = None):
    """Compute ODE integration error vs NFE (Figure 7 left).

    Generates reference solutions at high NFE, then compares lower-NFE solutions.
    """
    if nfe_list is None:
        nfe_list = [10, 20, 40, 60, 80, 100]

    print(f"Computing reference solutions at NFE={ref_nfe} for {n_seeds} seeds...")
    x0 = torch.randn(n_seeds, 3, image_size, image_size, device=device)
    ref = midpoint_sample(net, x0, nfe=ref_nfe)

    results = {}
    for nfe in nfe_list:
        for solver_name, solver_fn in SOLVERS.items():
            # Check NFE compatibility
            if solver_name == "midpoint" and nfe % 2 != 0:
                continue
            if solver_name == "rk4" and nfe % 4 != 0:
                continue

            samples = solver_fn(net, x0, nfe=nfe)
            mse = (samples - ref).pow(2).mean(dim=(1, 2, 3)).mean().item()
            key = f"{solver_name}_nfe{nfe}"
            results[key] = {"solver": solver_name, "nfe": nfe, "mse": mse}
            print(f"  {solver_name} NFE={nfe}: MSE={mse:.6f}")

    # Save results
    with open(os.path.join(out_dir, "ode_error_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for solver_name in SOLVERS:
        nfes = []
        mses = []
        for v in results.values():
            if v["solver"] == solver_name:
                nfes.append(v["nfe"])
                mses.append(v["mse"])
        if nfes:
            order = np.argsort(nfes)
            ax.plot(np.array(nfes)[order], np.array(mses)[order], "o-", label=solver_name)

    ax.set_xlabel("NFE")
    ax.set_ylabel("Per-pixel MSE")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("ODE Integration Error vs NFE")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ode_error_vs_nfe.png"), dpi=150)
    plt.close()
    print(f"Saved plot to {out_dir}/ode_error_vs_nfe.png")


def fid_vs_nfe_experiment(net, image_size: int, device: str, out_dir: str,
                          dataset: str, data_root: str,
                          n_samples: int = 50000, batch_size: int = 256,
                          nfe_list: list[int] | None = None):
    """Compute FID at various NFEs for each solver (Figure 7 right)."""
    if nfe_list is None:
        nfe_list = [10, 20, 40, 60, 80, 100]

    results = {}
    for solver_name in SOLVERS:
        for nfe in nfe_list:
            if solver_name == "midpoint" and nfe % 2 != 0:
                continue
            if solver_name == "rk4" and nfe % 4 != 0:
                continue

            print(f"\n{solver_name} NFE={nfe}: generating {n_samples} samples...")
            images, _ = generate_samples(
                net, n=n_samples, image_size=image_size, channels=3,
                device=device, solver=solver_name, nfe=nfe,
                atol=1e-5, rtol=1e-5, batch_size=batch_size,
            )

            sample_dir = os.path.join(out_dir, f"samples_{solver_name}_nfe{nfe}")
            save_images_to_dir(images, sample_dir)
            fid = compute_fid(sample_dir, dataset, data_root, image_size, device)

            key = f"{solver_name}_nfe{nfe}"
            results[key] = {"solver": solver_name, "nfe": nfe, "fid": fid}
            print(f"  {solver_name} NFE={nfe}: FID={fid:.2f}")

    # Save results
    with open(os.path.join(out_dir, "fid_vs_nfe_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for solver_name in SOLVERS:
        nfes = []
        fids = []
        for v in results.values():
            if v["solver"] == solver_name:
                nfes.append(v["nfe"])
                fids.append(v["fid"])
        if nfes:
            order = np.argsort(nfes)
            ax.plot(np.array(nfes)[order], np.array(fids)[order], "o-", label=solver_name)

    ax.set_xlabel("NFE")
    ax.set_ylabel("FID")
    ax.legend()
    ax.set_title("FID vs NFE")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fid_vs_nfe.png"), dpi=150)
    plt.close()
    print(f"Saved plot to {out_dir}/fid_vs_nfe.png")


def main():
    parser = argparse.ArgumentParser(description="Low-cost sampling experiments (Figure 7)")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, choices=["error", "fid", "both"])
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--n-samples", type=int, default=50000, help="Samples for FID")
    parser.add_argument("--n-seeds", type=int, default=256, help="Seeds for ODE error")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--nfe-list", type=int, nargs="+", default=[10, 20, 40, 60, 80, 100])
    parser.add_argument("--out-dir", type=str, default="./runs/low_cost")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    net, step, image_size = load_ckpt(args.ckpt, device=device)
    print(f"Loaded checkpoint step={step}, image_size={image_size}")

    if args.mode in ("error", "both"):
        ode_error_experiment(net, image_size, device, args.out_dir,
                             n_seeds=args.n_seeds, nfe_list=args.nfe_list)

    if args.mode in ("fid", "both"):
        fid_vs_nfe_experiment(net, image_size, device, args.out_dir,
                              args.dataset, args.data_root,
                              n_samples=args.n_samples, batch_size=args.batch_size,
                              nfe_list=args.nfe_list)


if __name__ == "__main__":
    main()
