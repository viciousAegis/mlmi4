"""Train and visualise 2D flow matching on checkerboard data (Figures 4, 9)."""
from __future__ import annotations
import os
import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm

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
    t = torch.rand(B, device=x1.device) * (1.0 - 2 * eps_t) + eps_t
    eps = torch.randn_like(x1)
    t_req = t.detach().clone().requires_grad_(True)
    # Time-reversed: evaluate alpha_bar at (1-t) per paper Eq. 18
    s = 1.0 - t_req
    log_alpha = -0.5 * (beta_min * s + 0.5 * (beta_max - beta_min) * s ** 2)
    alpha = torch.exp(log_alpha)
    mu_scale = alpha
    sig = torch.sqrt(1.0 - alpha ** 2)
    mu_t = mu_scale[:, None] * x1
    x_t = mu_t + sig[:, None] * eps
    dmu_scale = torch.autograd.grad(mu_scale.sum(), t_req, retain_graph=True)[0]
    dsig = torch.autograd.grad(sig.sum(), t_req)[0]
    dmu = dmu_scale[:, None] * x1
    u_t = dmu + (dsig[:, None] / sig[:, None]) * (x_t - mu_t)
    return t.detach(), x_t.detach(), u_t.detach()


# -- Training ---------------------------------------------------------------

def train(path_type: str, steps: int = 20000, lr: float = 1e-3, batch: int = 4096,
          device: str = "cpu", sigma_min: float = 0.01, hidden: int = 512, num_layers: int = 5):
    net = FlowMatchingMLP(hidden=hidden, num_layers=num_layers).to(device)
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
    """Fixed-step midpoint ODE solver for 2D flow.

    nfe = total number of function evaluations (must be even).
    Each midpoint step uses 2 evaluations.
    """
    assert nfe % 2 == 0, "Midpoint method requires even NFE"
    n_steps = nfe // 2
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_i = i * dt
        t_mid = t_i + 0.5 * dt
        t_vec = torch.full((n,), t_i, device=device)
        k1 = net(x, t_vec)
        t_mid_vec = torch.full((n,), t_mid, device=device)
        k2 = net(x + 0.5 * dt * k1, t_mid_vec)
        x = x + dt * k2
    return x


# -- Visualisation -----------------------------------------------------------

def _plot_density_panel(pts, out_path, title, bins=200, cmap="magma"):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.hist2d(
        pts[:, 0],
        pts[:, 1],
        bins=bins,
        range=[[-5, 5], [-5, 5]],
        cmap=cmap,
        density=True,
        norm=PowerNorm(gamma=0.7),
        cmin=1e-9,
    )
    ax.set_facecolor("white")
    ax.set_title(title)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_density(net, path_type, nfe_list, out_dir, device="cpu"):
    """Generate one density plot file per NFE."""
    for nfe in nfe_list:
        samples = ode_sample(net, n=20000, nfe=nfe, device=device).cpu().numpy()
        out_path = os.path.join(out_dir, f"checkerboard_density_{path_type}_nfe{nfe}.png")
        _plot_density_panel(samples, out_path, title=f"NFE={nfe}")


def plot_trajectories(net, path_type, out_dir, n_traj=200, nfe=200, device="cpu"):
    """Visualise ODE trajectories from noise to data (Figure 9 style)."""
    assert nfe % 2 == 0
    n_steps = nfe // 2
    x = torch.randn(n_traj, 2, device=device)
    trajectory = [x.cpu().numpy()]
    dt = 1.0 / n_steps

    with torch.no_grad():
        for i in range(n_steps):
            t_i = i * dt
            t_vec = torch.full((n_traj,), t_i, device=device)
            t_mid = t_i + 0.5 * dt
            k1 = net(x, t_vec)
            t_mid_vec = torch.full((n_traj,), t_mid, device=device)
            k2 = net(x + 0.5 * dt * k1, t_mid_vec)
            x = x + dt * k2
            trajectory.append(x.cpu().numpy())

    trajectory = np.array(trajectory)  # (n_steps+1, n_traj, 2)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for j in range(n_traj):
        ax.plot(trajectory[:, j, 0], trajectory[:, j, 1], alpha=0.15, lw=0.5, color="blue")
    ax.scatter(trajectory[-1, :, 0], trajectory[-1, :, 1], s=1, color="red", alpha=0.5)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.set_title(f"FM w/ {path_type.upper()} — Trajectories")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"checkerboard_traj_{path_type}.png"), dpi=150)
    plt.close()


def plot_time_snapshots(net, path_type, out_dir, n=20000, nfe=200, device="cpu"):
    """Show density snapshots as separate files."""
    assert nfe % 2 == 0
    n_steps = nfe // 2
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / n_steps
    snapshots = {0: x.cpu().numpy()}
    snapshot_ids = np.linspace(0, n_steps, 4, dtype=int).tolist()

    with torch.no_grad():
        for i in range(n_steps):
            t_i = i * dt
            t_vec = torch.full((n,), t_i, device=device)
            t_mid = t_i + 0.5 * dt
            k1 = net(x, t_vec)
            t_mid_vec = torch.full((n,), t_mid, device=device)
            k2 = net(x + 0.5 * dt * k1, t_mid_vec)
            x = x + dt * k2
            if (i + 1) in snapshot_ids:
                snapshots[i + 1] = x.cpu().numpy()

    for snap_idx, step_id in enumerate(snapshot_ids):
        pts = snapshots[step_id]
        t_val = step_id / n_steps
        out_path = os.path.join(out_dir, f"checkerboard_snapshots_{path_type}_t{snap_idx}.png")
        _plot_density_panel(pts, out_path, title=rf"$t={t_val:.2f}$", bins=200, cmap="magma")


def _collect_snapshots(net, n=20000, nfe=200, device="cpu", n_panels=10):
    """Collect point cloud snapshots from t=0 to t=1 for a fixed-step midpoint solve."""
    assert nfe % 2 == 0
    n_steps = nfe // 2
    step_ids = np.linspace(0, n_steps, n_panels, dtype=int).tolist()
    x = torch.randn(n, 2, device=device)
    dt = 1.0 / n_steps
    snapshots = {0: x.cpu().numpy()}

    with torch.no_grad():
        for i in range(n_steps):
            t_i = i * dt
            t_vec = torch.full((n,), t_i, device=device)
            t_mid = t_i + 0.5 * dt
            k1 = net(x, t_vec)
            t_mid_vec = torch.full((n,), t_mid, device=device)
            k2 = net(x + 0.5 * dt * k1, t_mid_vec)
            x = x + dt * k2
            if (i + 1) in step_ids:
                snapshots[i + 1] = x.cpu().numpy()

    return [snapshots[sid] for sid in step_ids]


def plot_figure4_style(nets, out_dir, device="cpu"):
    """Create a Figure-4-style composite: time trajectories (left) and NFE sweep (right)."""
    left_cols = 10
    right_nfes = [2, 4, 8, 10]
    fig, axes = plt.subplots(
        2,
        left_cols + len(right_nfes),
        figsize=(24, 7),
        gridspec_kw={"width_ratios": [1] * left_cols + [1.2] * len(right_nfes)},
    )

    row_info = [("diffusion", "FM w/ Diffusion"), ("ot", "FM w/ OT")]
    for row, (path_type, row_title) in enumerate(row_info):
        net = nets[path_type]
        snapshots = _collect_snapshots(net, n=15000, nfe=200, device=device, n_panels=left_cols)
        for col in range(left_cols):
            ax = axes[row, col]
            pts = snapshots[col]
            ax.hist2d(
                pts[:, 0],
                pts[:, 1],
                bins=170,
                range=[[-5, 5], [-5, 5]],
                cmap="magma",
                density=True,
                norm=PowerNorm(gamma=0.7),
                cmin=1e-9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"t={col/(left_cols-1):.2f}", fontsize=9)

        for idx, nfe in enumerate(right_nfes):
            ax = axes[row, left_cols + idx]
            samples = ode_sample(net, n=20000, nfe=nfe, device=device).cpu().numpy()
            ax.hist2d(
                samples[:, 0],
                samples[:, 1],
                bins=170,
                range=[[-5, 5], [-5, 5]],
                cmap="magma",
                density=True,
                norm=PowerNorm(gamma=0.7),
                cmin=1e-9,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_aspect("equal")
            if row == 1:
                ax.set_xlabel(f"NFE={nfe}", fontsize=12)

        axes[row, 0].set_ylabel(row_title, fontsize=12)

    fig.suptitle("Checkerboard FM: Diffusion vs OT", fontsize=16)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "checkerboard_figure4_style.png")
    plt.savefig(out_path, dpi=180)
    plt.close()


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="2D checkerboard flow matching experiment")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--make-composite", action="store_true", help="Also save combined Figure-4-style panel grid.")
    parser.add_argument("--out-dir", type=str, default="./runs/checkerboard")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    nets = {}
    for path_type in ["ot", "diffusion"]:
        print(f"\n{'='*60}")
        print(f"Training FM w/ {path_type.upper()}")
        print(f"{'='*60}")
        net = train(
            path_type,
            steps=args.steps,
            lr=args.lr,
            batch=args.batch,
            device=device,
            hidden=args.hidden,
            num_layers=args.num_layers,
        )
        nets[path_type] = net

        # Save model
        torch.save(net.state_dict(), os.path.join(args.out_dir, f"mlp_{path_type}.pt"))

        # Visualisations
        plot_density(net, path_type, nfe_list=[2, 4, 8, 10], out_dir=args.out_dir, device=device)
        plot_trajectories(net, path_type, out_dir=args.out_dir, device=device)
        plot_time_snapshots(net, path_type, out_dir=args.out_dir, device=device)

    if args.make_composite:
        plot_figure4_style(nets, out_dir=args.out_dir, device=device)

    print(f"\nAll outputs saved to {args.out_dir}")


if __name__ == "__main__":
    main()
