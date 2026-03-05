"""Evaluate negative log-likelihood (NLL) in bits per dimension (BPD).

Uses the instantaneous change of variables formula (Eq. 35-37 in the paper):
    log p_1(x_1) = log p_0(x_0) - integral_0^1 div(v_t(x_t)) dt

The divergence is estimated via the Hutchinson trace estimator.
Supports importance-weighted NLL with uniform dequantization (Table 4, Eq. 47).

Usage:
    python evaluate_nll.py --ckpt runs/fm_ot_cifar10/ckpt.pt --dataset cifar10
    python evaluate_nll.py --ckpt runs/fm_ot_cifar10/ckpt.pt --dataset cifar10 --K 10
"""
from __future__ import annotations
import argparse
import math

import torch
import torch.nn as nn
import numpy as np

from sample import load_ckpt
from src.datasets import get_dataloaders


def hutchinson_divergence(net: nn.Module, x: torch.Tensor, t: torch.Tensor,
                          noise: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute v_t(x) and Hutchinson estimate of div(v_t(x)).

    Args:
        net: vector field network
        x: (B, C, H, W) current state
        t: (B,) time
        noise: (B, C, H, W) Rademacher noise for trace estimation

    Returns:
        v: (B, C, H, W) vector field
        div_v: (B,) estimated divergence
    """
    x = x.detach().requires_grad_(True)
    v = net(x, t)
    # Compute (noise^T @ Jv @ noise) = trace estimator
    # via vjp: (d/dx)(v . noise) gives Jv^T @ noise, then dot with noise
    vn = (v * noise).sum()
    grad_vn = torch.autograd.grad(vn, x, create_graph=False)[0]
    div_v = (grad_vn * noise).flatten(start_dim=1).sum(dim=1)
    return v.detach(), div_v.detach()


def compute_log_likelihood(net: nn.Module, x1: torch.Tensor, device: str,
                           atol: float = 1e-5, rtol: float = 1e-5) -> torch.Tensor:
    """Compute log p_1(x_1) for a batch using dopri5 + Hutchinson trace estimator.

    Integrates the augmented ODE backwards from t=1 to t=0:
        dx/dt = v_t(x)
        d(log_p)/dt = -div(v_t(x))

    Returns:
        log_p1: (B,) log-likelihoods
    """
    from torchdiffeq import odeint

    B = x1.shape[0]
    noise = torch.randint(0, 2, x1.shape, device=device).float() * 2 - 1  # Rademacher

    # Augmented state: [x, log_p_change]
    # We integrate from t=1 backward to t=0
    state0 = torch.cat([x1.reshape(B, -1), torch.zeros(B, 1, device=device)], dim=1)
    flat_dim = x1.shape[1] * x1.shape[2] * x1.shape[3]
    noise_flat = noise.reshape(B, -1)

    class AugmentedVF(nn.Module):
        def forward(self, t, state):
            x_flat = state[:, :flat_dim]
            x = x_flat.reshape(x1.shape)
            t_batch = torch.full((B,), float(t), device=device, dtype=x.dtype)
            v, div_v = hutchinson_divergence(net, x, t_batch,
                                             noise.to(x.dtype))
            return torch.cat([v.reshape(B, -1), -div_v[:, None]], dim=1)

    vf = AugmentedVF().to(device)
    # Integrate backward: t=1 -> t=0
    ts = torch.tensor([1.0, 0.0], device=device)

    with torch.no_grad():
        result = odeint(vf, state0, ts, method="dopri5", atol=atol, rtol=rtol)

    final_state = result[-1]
    x0 = final_state[:, :flat_dim]
    log_p_change = final_state[:, -1]  # integral of -div(v)

    # log p_0(x_0) under standard normal
    log_p0 = -0.5 * (x0.pow(2).sum(dim=1) + flat_dim * math.log(2 * math.pi))

    # log p_1(x_1) = log p_0(x_0) + integral_0^1 (-div v_t) dt
    log_p1 = log_p0 + log_p_change
    return log_p1


def nll_with_dequantization(net: nn.Module, x_int: torch.Tensor, device: str,
                            K: int = 1, atol: float = 1e-5, rtol: float = 1e-5) -> torch.Tensor:
    """NLL with uniform dequantization and optional importance weighting (Eq. 47).

    Args:
        x_int: (B, C, H, W) integer pixel values in [0, 255]
        K: number of importance samples

    Returns:
        nll: (B,) negative log-likelihood in nats
    """
    B = x_int.shape[0]

    log_probs = []
    for _ in range(K):
        # Uniform dequantization: add U[0,1)/256 noise then map to [-1,1]
        u = torch.rand_like(x_int.float())
        x_deq = (x_int.float() + u) / 256.0  # [0, 1]
        x_deq = x_deq * 2.0 - 1.0  # [-1, 1]

        log_p = compute_log_likelihood(net, x_deq, device, atol=atol, rtol=rtol)
        # Jacobian of transform: d/dx (2x/256 - 1) = 2/256 = 1/128 per dimension
        # Total log |det J| = D * log(1/128) = -D * log(128)
        D = x_int.shape[1] * x_int.shape[2] * x_int.shape[3]
        log_p = log_p - D * math.log(128.0)
        log_probs.append(log_p)

    # Importance-weighted estimate: log(1/K * sum exp(log_p_k))
    log_probs = torch.stack(log_probs, dim=0)  # (K, B)
    log_p_iw = torch.logsumexp(log_probs, dim=0) - math.log(K)
    return -log_p_iw


def nats_to_bpd(nll_nats: float, dim: int) -> float:
    """Convert nats to bits per dimension."""
    return nll_nats / (dim * math.log(2))


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLL/BPD for flow matching model")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--K", type=int, default=1, help="Importance weighting samples")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Limit evaluation to N batches (for quick testing)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    net, step, image_size = load_ckpt(args.ckpt, device=device)
    print(f"Loaded checkpoint step={step}, image_size={image_size}")

    loaders = get_dataloaders(
        dataset=args.dataset, root=args.data_root, batch_size=args.batch_size,
        val_size=1, num_workers=4, image_size=image_size,
    )
    dim = 3 * image_size * image_size

    all_nll = []
    for i, (batch, _) in enumerate(loaders.test):
        if args.max_batches is not None and i >= args.max_batches:
            break

        # Convert from [-1,1] back to [0,255] integer for dequantization
        x_int = ((batch.clamp(-1, 1) + 1) * 0.5 * 255).round().clamp(0, 255)
        x_int = x_int.to(device)

        nll = nll_with_dequantization(net, x_int, device, K=args.K,
                                       atol=args.atol, rtol=args.rtol)
        all_nll.append(nll.cpu())

        avg_bpd = nats_to_bpd(torch.cat(all_nll).mean().item(), dim)
        print(f"Batch {i+1}: running BPD = {avg_bpd:.4f}")

    all_nll = torch.cat(all_nll)
    mean_nll = all_nll.mean().item()
    mean_bpd = nats_to_bpd(mean_nll, dim)
    std_bpd = nats_to_bpd(all_nll.std().item() / math.sqrt(len(all_nll)), dim)

    print(f"\n{'='*50}")
    print(f"NLL (nats): {mean_nll:.2f}")
    print(f"BPD: {mean_bpd:.4f} +/- {std_bpd:.4f}")
    print(f"K={args.K}, n_samples={len(all_nll)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
