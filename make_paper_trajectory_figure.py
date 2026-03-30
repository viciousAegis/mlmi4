"""Create a paper-ready trajectory comparison figure for checkerboard FM models."""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from src.checkerboard import FlowMatchingMLP


@torch.no_grad()
def integrate_trajectories(net: torch.nn.Module, x0: torch.Tensor, nfe: int = 200) -> np.ndarray:
    """Midpoint integration from t=0 to t=1. Returns (steps+1, n, 2)."""
    assert nfe % 2 == 0, "Midpoint method requires even NFE"
    n_steps = nfe // 2
    dt = 1.0 / n_steps
    x = x0.clone()
    traj = [x.cpu().numpy()]

    for i in range(n_steps):
        t_i = i * dt
        t_mid = t_i + 0.5 * dt
        t_vec = torch.full((x.shape[0],), t_i, device=x.device)
        t_mid_vec = torch.full((x.shape[0],), t_mid, device=x.device)
        k1 = net(x, t_vec)
        k2 = net(x + 0.5 * dt * k1, t_mid_vec)
        x = x + dt * k2
        traj.append(x.cpu().numpy())

    return np.array(traj)


def add_trajectory_panel(
    ax,
    traj: np.ndarray,
    title: str,
    line_cmap: str = "magma",
    panel_bg: str = "#090909",
    title_color: str = "white",
    spine_color: str = "#8A8A8A",
    start_color: str = "#ffffff",
    end_color: str = "#FDE725",
    show_density: bool = True,
) -> None:
    """Render trajectory panel with time-colored flow lines and endpoints."""
    t_steps, n_traj, _ = traj.shape
    n_segments = t_steps - 1

    rng = np.random.default_rng(0)
    vis_idx = rng.choice(n_traj, size=min(260, n_traj), replace=False)

    start = traj[:-1, vis_idx, :]  # (S, M, 2)
    end = traj[1:, vis_idx, :]     # (S, M, 2)
    segments = np.stack([start, end], axis=2).reshape(-1, 2, 2)
    color_t = np.repeat(np.linspace(0.0, 1.0, n_segments), len(vis_idx))

    if show_density:
        # Background density at t=1 to make the target geometry visually explicit.
        x_end_bg = traj[-1]
        ax.hist2d(
            x_end_bg[:, 0],
            x_end_bg[:, 1],
            bins=180,
            range=[[-5, 5], [-5, 5]],
            cmap="magma",
            density=True,
            alpha=0.50,
        )

    lc = LineCollection(
        segments,
        cmap=line_cmap,
        norm=Normalize(vmin=0.0, vmax=1.0),
        linewidths=0.65,
        alpha=0.55,
    )
    lc.set_array(color_t)
    ax.add_collection(lc)

    x_start = traj[0]
    x_end = traj[-1]
    ax.scatter(x_start[:, 0], x_start[:, 1], s=1.0, c=start_color, alpha=0.06, linewidths=0, zorder=2)
    ax.scatter(x_end[:, 0], x_end[:, 1], s=1.6, c=end_color, alpha=0.35, linewidths=0, zorder=3)

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect("equal")
    ax.set_facecolor(panel_bg)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, color=title_color, pad=10)

    for spine in ax.spines.values():
        spine.set_color(spine_color)
        spine.set_linewidth(0.9)


def load_model(ckpt_path: str, device: str, hidden: int, num_layers: int) -> torch.nn.Module:
    net = FlowMatchingMLP(hidden=hidden, num_layers=num_layers).to(device)
    state = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-ready checkerboard trajectory figure")
    parser.add_argument("--ckpt-dir", type=str, default="runs/checkerboard_magma_final")
    parser.add_argument("--out-dir", type=str, default="runs/checkerboard_magma_final")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--n-traj", type=int, default=900)
    parser.add_argument("--nfe", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ot_path = os.path.join(args.ckpt_dir, "mlp_ot.pt")
    diff_path = os.path.join(args.ckpt_dir, "mlp_diffusion.pt")
    ot_net = load_model(ot_path, device, args.hidden, args.num_layers)
    diff_net = load_model(diff_path, device, args.hidden, args.num_layers)

    x0 = torch.randn(args.n_traj, 2, device=device)
    ot_traj = integrate_trajectories(ot_net, x0, nfe=args.nfe)
    diff_traj = integrate_trajectories(diff_net, x0, nfe=args.nfe)

    theme_specs = [
        {
            "suffix": "",
            "fig_bg": "#050505",
            "panel_bg": "#090909",
            "title_color": "white",
            "spine_color": "#8A8A8A",
            "start_color": "#ffffff",
            "end_color": "#FDE725",
            "suptitle_color": "white",
            "show_density": True,
        },
        {
            "suffix": "_whitebg",
            "fig_bg": "#ffffff",
            "panel_bg": "#ffffff",
            "title_color": "#111111",
            "spine_color": "#444444",
            "start_color": "#444444",
            "end_color": "#2A6F97",
            "suptitle_color": "#111111",
            "show_density": False,
        },
    ]

    for theme in theme_specs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        fig.patch.set_facecolor(theme["fig_bg"])
        add_trajectory_panel(
            axes[0],
            diff_traj,
            "Flow Matching with Diffusion",
            panel_bg=theme["panel_bg"],
            title_color=theme["title_color"],
            spine_color=theme["spine_color"],
            start_color=theme["start_color"],
            end_color=theme["end_color"],
            show_density=theme["show_density"],
        )
        add_trajectory_panel(
            axes[1],
            ot_traj,
            "Flow Matching with OT",
            panel_bg=theme["panel_bg"],
            title_color=theme["title_color"],
            spine_color=theme["spine_color"],
            start_color=theme["start_color"],
            end_color=theme["end_color"],
            show_density=theme["show_density"],
        )

        fig.suptitle(
            "Checkerboard CNF Trajectories (Midpoint Solver, NFE=200)",
            fontsize=18,
            color=theme["suptitle_color"],
            y=1.02,
        )

        png_path = os.path.join(args.out_dir, f"paper_trajectory_comparison{theme['suffix']}.png")
        pdf_path = os.path.join(args.out_dir, f"paper_trajectory_comparison{theme['suffix']}.pdf")
        fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved {png_path}")
        print(f"Saved {pdf_path}")

        single_specs = [
            (diff_traj, "Flow Matching with Diffusion", f"paper_trajectory_diffusion{theme['suffix']}"),
            (ot_traj, "Flow Matching with OT", f"paper_trajectory_ot{theme['suffix']}"),
        ]
        for traj, title, stem in single_specs:
            fig1, ax1 = plt.subplots(1, 1, figsize=(6.4, 6.0), constrained_layout=True)
            fig1.patch.set_facecolor(theme["fig_bg"])
            add_trajectory_panel(
                ax1,
                traj,
                title,
                panel_bg=theme["panel_bg"],
                title_color=theme["title_color"],
                spine_color=theme["spine_color"],
                start_color=theme["start_color"],
                end_color=theme["end_color"],
                show_density=theme["show_density"],
            )
            png1 = os.path.join(args.out_dir, f"{stem}.png")
            pdf1 = os.path.join(args.out_dir, f"{stem}.pdf")
            fig1.savefig(png1, dpi=320, bbox_inches="tight", facecolor=fig1.get_facecolor())
            fig1.savefig(pdf1, bbox_inches="tight", facecolor=fig1.get_facecolor())
            plt.close(fig1)
            print(f"Saved {png1}")
            print(f"Saved {pdf1}")


if __name__ == "__main__":
    main()
