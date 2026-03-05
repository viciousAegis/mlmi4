"""Train and visualise 2D flow matching on checkerboard data (Figures 4, 9)."""
from __future__ import annotations
import os
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from src.checkerboard import sample_checkerboard, FlowMatchingMLP
from src.paths import ot_path_and_target, diffusion_path_and_target


# -- 2D path helpers (work on (B,2) tensors instead of (B,C,H,W)) --------

def ot_path_2d(x1: torch.Tensor, sigma_min: float = 0.01):
    B = x1.shape[0]
    t = torch.rand(B, device=x1.device)
    eps = torch.randn_like(x1)
    t_ = t[:, None]
    mu_t = t_ * x1
    sigma_t = (1.0 - t_) + t_ * sigma_min
    x_t = mu_t + sigma_t * eps
    dmu = x1
    dsigma = sigma_min - 1.0
    u_t = dmu + (dsigma / sigma_t) * (x_t - mu_t)
    return t, x_t, u_t


def diffusion_path_2d(x1: torch.Tensor, beta_min: float = 0.1, beta_max: float = 20.0, eps_t: float = 1e-5):
    B = x1.shape[0]
    t = torch.rand(B, device=x1.device) * (1.0 - eps_t)
    eps = torch.randn_like(x1)
    t_req = t.detach().clone().requires_grad_(True)
    log_abar = -0.5 * (beta_min * t_req + 0.5 * (beta_max - beta_min) * t_req ** 2)
    abar = torch.exp(log_abar)
    mu_scale = torch.sqrt(abar)
    sig = torch.sqrt(1.0 - abar)
    mu_t = mu_scale[:, None] * x1
    x_t = mu_t + sig[:, None] * eps
    dmu_scale = torch.autograd.grad(mu_scale.sum(), t_req, retain_graph=True)[0]
    dsig = torch.autograd.grad(sig.sum(), t_req)[0]
    dmu = dmu_scale[:, None] * x1
    u_t = dmu + (dsig[:, None] / sig[:, None]) * (x_t - mu_t)
    return t.detach(), x_t.detach(), u_t.detach()


# -- Training ---------------------------------------------------------------

def train(path_type: str, steps: int = 20000, lr: float = 1e-3, batch: int = 4096,
          device: str = "cpu", sigma_min: float = 0.01):
    net = FlowMatchingMLP().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    for step in range(1, steps + 1):
        x1 = sample_checkerboard(batch).to(device)
        if path_type == "ot":
            t, x_t, u_t = ot_path_2d(x1, sigma_min=sigma_min)
        else:
            t, x_t, u_t = diffusion_path_2d(x1)

        v = net(x_t, t)
        loss = (v - u_t).pow(2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 2000 == 0 or step == 1:
            print(f"[{path_type}] step {step:5d} | loss {loss.item():.6f}")

    return net


# -- ODE sampling -----------------------------------------------------------

@torch.no_grad()
def ode_sample(net: nn.Module, n: int, nfe: int, device: str = "cpu"):
    """Fixed-step midpoint ODE solver for 2D flow."""
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / nfe
    for i in range(nfe):
        t_i = i * dt
        t_mid = t_i + 0.5 * dt
        # midpoint step
        t_vec = torch.full((n,), t_i, device=device)
        k1 = net(x, t_vec)
        t_mid_vec = torch.full((n,), t_mid, device=device)
        k2 = net(x + 0.5 * dt * k1, t_mid_vec)
        x = x + dt * k2
    return x


# -- Visualisation -----------------------------------------------------------

def plot_density(net, path_type, nfe_list, out_dir, device="cpu"):
    """Generate samples at different NFEs and plot density (Figure 4 style)."""
    fig, axes = plt.subplots(1, len(nfe_list), figsize=(4 * len(nfe_list), 4))
    if len(nfe_list) == 1:
        axes = [axes]

    for ax, nfe in zip(axes, nfe_list):
        samples = ode_sample(net, n=20000, nfe=nfe, device=device).cpu().numpy()
        ax.hist2d(samples[:, 0], samples[:, 1], bins=200, range=[[-5, 5], [-5, 5]],
                  cmap="inferno", density=True)
        ax.set_title(f"NFE={nfe}")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")

    fig.suptitle(f"FM w/ {path_type.upper()} — Generated Density", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"checkerboard_density_{path_type}.png"), dpi=150)
    plt.close()


def plot_trajectories(net, path_type, out_dir, n_traj=200, nfe=100, device="cpu"):
    """Visualise ODE trajectories from noise to data (Figure 9 style)."""
    x = torch.randn(n_traj, 2, device=device)
    trajectory = [x.cpu().numpy()]
    dt = 1.0 / nfe

    with torch.no_grad():
        for i in range(nfe):
            t_i = i * dt
            t_vec = torch.full((n_traj,), t_i, device=device)
            t_mid = t_i + 0.5 * dt
            k1 = net(x, t_vec)
            t_mid_vec = torch.full((n_traj,), t_mid, device=device)
            k2 = net(x + 0.5 * dt * k1, t_mid_vec)
            x = x + dt * k2
            trajectory.append(x.cpu().numpy())

    trajectory = np.array(trajectory)  # (nfe+1, n_traj, 2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for j in range(n_traj):
        ax.plot(trajectory[:, j, 0], trajectory[:, j, 1], alpha=0.15, lw=0.5, color="blue")
    ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], s=1, color="red", alpha=0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_title(f"FM w/ {path_type.upper()} — Trajectories")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"checkerboard_traj_{path_type}.png"), dpi=150)
    plt.close()


def plot_time_snapshots(net, path_type, out_dir, n=20000, nfe=200, device="cpu"):
    """Show density at t=0, 1/3, 2/3, 1 (Figure 4 style)."""
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / nfe
    snapshots = {0: x.cpu().numpy()}
    snapshot_times = {int(nfe / 3): "t=1/3", int(2 * nfe / 3): "t=2/3", nfe: "t=1"}

    with torch.no_grad():
        for i in range(nfe):
            t_i = i * dt
            t_vec = torch.full((n,), t_i, device=device)
            t_mid = t_i + 0.5 * dt
            k1 = net(x, t_vec)
            t_mid_vec = torch.full((n,), t_mid, device=device)
            k2 = net(x + 0.5 * dt * k1, t_mid_vec)
            x = x + dt * k2
            if (i + 1) in snapshot_times:
                snapshots[i + 1] = x.cpu().numpy()

    times = sorted(snapshots.keys())
    labels = ["t=0"] + [snapshot_times[t] for t in times[1:]]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, t_idx, label in zip(axes, times, labels):
        pts = snapshots[t_idx]
        ax.hist2d(pts[:, 0], pts[:, 1], bins=200, range=[[-5, 5], [-5, 5]],
                  cmap="inferno", density=True)
        ax.set_title(label)
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect("equal")

    fig.suptitle(f"FM w/ {path_type.upper()} — Density Evolution", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"checkerboard_snapshots_{path_type}.png"), dpi=150)
    plt.close()


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="2D checkerboard flow matching experiment")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--out-dir", type=str, default="./runs/checkerboard")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    for path_type in ["ot", "diffusion"]:
        print(f"\n{'='*60}")
        print(f"Training FM w/ {path_type.upper()}")
        print(f"{'='*60}")
        net = train(path_type, steps=args.steps, lr=args.lr, batch=args.batch, device=device)

        # Save model
        torch.save(net.state_dict(), os.path.join(args.out_dir, f"mlp_{path_type}.pt"))

        # Visualisations
        plot_density(net, path_type, nfe_list=[4, 8, 10, 20, 100], out_dir=args.out_dir, device=device)
        plot_trajectories(net, path_type, out_dir=args.out_dir, device=device)
        plot_time_snapshots(net, path_type, out_dir=args.out_dir, device=device)

    print(f"\nAll outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
