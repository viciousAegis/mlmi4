"""Evaluate FID for a trained flow matching model.

Usage:
    python evaluate_fid.py --ckpt runs/fm_ot_cifar10/ckpt.pt --dataset cifar10 --n 50000
    python evaluate_fid.py --ckpt runs/fm_ot_cifar10/ckpt.pt --dataset cifar10 --solver midpoint --nfe 20

Requires: pip install pytorch-fid
The first run will compute and cache dataset statistics.
"""
from __future__ import annotations
import os
import argparse
import tempfile

import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image

from sample import load_ckpt, to_img
from src.solvers import SOLVERS, dopri5_sample


def generate_samples(net, n: int, image_size: int, channels: int, device: str,
                     solver: str, nfe: int | None, atol: float, rtol: float,
                     batch_size: int = 256):
    """Generate n samples and return as uint8 numpy array (n, H, W, C)."""
    net.eval()
    all_samples = []
    total_nfe = 0
    n_batches = 0

    remaining = n
    while remaining > 0:
        bs = min(batch_size, remaining)
        x0 = torch.randn(bs, channels, image_size, image_size, device=device)

        if solver == "dopri5":
            samples, nfe_count = dopri5_sample(net, x0, atol=atol, rtol=rtol)
            total_nfe += nfe_count
        else:
            solver_fn = SOLVERS[solver]
            samples = solver_fn(net, x0, nfe=nfe)
            total_nfe += nfe

        n_batches += 1
        # Convert to [0, 255] uint8
        imgs = to_img(samples)  # [0, 1]
        imgs = (imgs * 255).clamp(0, 255).to(torch.uint8)
        imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
        all_samples.append(imgs)
        remaining -= bs

        if n_batches % 10 == 0:
            print(f"  Generated {n - remaining}/{n} samples...")

    avg_nfe = total_nfe / n_batches
    return np.concatenate(all_samples, axis=0)[:n], avg_nfe


def save_images_to_dir(images: np.ndarray, out_dir: str):
    """Save numpy images to directory as individual PNGs for pytorch-fid."""
    os.makedirs(out_dir, exist_ok=True)
    for i, img in enumerate(images):
        Image.fromarray(img).save(os.path.join(out_dir, f"{i:06d}.png"))


def compute_fid(sample_dir: str, dataset: str, data_root: str, image_size: int,
                device: str) -> float:
    """Compute FID using pytorch-fid between generated samples and dataset."""
    from pytorch_fid import fid_score

    # Get or create real image stats directory
    real_dir = os.path.join(data_root, f"{dataset}_{image_size}_real_images")

    if not os.path.exists(real_dir) or len(os.listdir(real_dir)) == 0:
        print(f"Extracting real images to {real_dir}...")
        _extract_real_images(dataset, data_root, image_size, real_dir)

    print("Computing FID...")
    fid = fid_score.calculate_fid_given_paths(
        [real_dir, sample_dir],
        batch_size=256,
        device=device,
        dims=2048,
    )
    return fid


def _extract_real_images(dataset: str, data_root: str, image_size: int, out_dir: str):
    """Extract real dataset images as PNGs for FID computation."""
    from src.datasets import get_dataloaders

    os.makedirs(out_dir, exist_ok=True)
    loaders = get_dataloaders(dataset=dataset, root=data_root, batch_size=256,
                              val_size=1, num_workers=4, image_size=image_size)

    idx = 0
    for batch, _ in loaders.train:
        for img in batch:
            img = (img.clamp(-1, 1) + 1) * 0.5  # [-1,1] -> [0,1]
            img = (img * 255).clamp(0, 255).to(torch.uint8)
            img = img.permute(1, 2, 0).numpy()
            Image.fromarray(img).save(os.path.join(out_dir, f"{idx:06d}.png"))
            idx += 1

    print(f"Saved {idx} real images to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FID for flow matching model")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--n", type=int, default=50000, help="Number of samples for FID")
    parser.add_argument("--solver", type=str, default="dopri5",
                        choices=["dopri5", "euler", "midpoint", "rk4"])
    parser.add_argument("--nfe", type=int, default=100, help="NFE for fixed-step solvers")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to save generated images")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    net, step, image_size = load_ckpt(args.ckpt, device=device)
    print(f"Loaded checkpoint step={step}, image_size={image_size}")

    print(f"Generating {args.n} samples with solver={args.solver}...")
    images, avg_nfe = generate_samples(
        net, n=args.n, image_size=image_size, channels=3, device=device,
        solver=args.solver, nfe=args.nfe, atol=args.atol, rtol=args.rtol,
        batch_size=args.batch_size,
    )
    print(f"Average NFE: {avg_nfe:.1f}")

    # Save to temp or specified directory
    sample_dir = args.out_dir or tempfile.mkdtemp(prefix="fm_samples_")
    print(f"Saving samples to {sample_dir}...")
    save_images_to_dir(images, sample_dir)

    fid = compute_fid(sample_dir, args.dataset, args.data_root, image_size, device)
    print(f"\n{'='*50}")
    print(f"FID ({args.solver}, NFE={avg_nfe:.0f}): {fid:.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
