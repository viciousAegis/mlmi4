# sample.py
from __future__ import annotations
import os
import argparse
import torch
from torchvision.utils import save_image, make_grid

from src.models import UNetCIFAR


def load_ckpt(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    net = UNetCIFAR().to(device)
    net.load_state_dict(ckpt["model"], strict=True)

    ema_state = ckpt.get("ema", None)
    if ema_state is not None:
        # Use EMA weights for sampling (recommended)
        net.load_state_dict(ema_state, strict=True)
        print("Loaded EMA weights for sampling.")
    else:
        print("No EMA in checkpoint; using raw model weights.")

    step = ckpt.get("step", None)
    return net, step


@torch.no_grad()
def dopri5_sample(
    net: UNetCIFAR, n: int = 64, device: str = "cuda", atol=1e-5, rtol=1e-5, return_trajectory=False, num_steps=10
):
    """
    Paper-style ODE solver using torchdiffeq.
    Install: pip install torchdiffeq
    
    Args:
        return_trajectory: if True, return states at multiple time points
        num_steps: number of time points to save (only used if return_trajectory=True)
    """
    from torchdiffeq import odeint

    net.eval()
    x0 = torch.randn(n, 3, 32, 32, device=device)

    class VF(torch.nn.Module):
        def forward(self, t, x):
            # torchdiffeq passes scalar t; expand to batch tensor
            t_batch = torch.full(
                (x.shape[0],), float(t), device=x.device, dtype=x.dtype
            )
            return net(x, t_batch)

    vf = VF().to(device)
    
    if return_trajectory:
        # Return states at multiple time points
        ts = torch.linspace(0.0, 1.0, num_steps, device=device)
        trajectory = odeint(vf, x0, ts, method="dopri5", atol=atol, rtol=rtol)
        return trajectory  # shape: (num_steps, n, 3, 32, 32)
    else:
        ts = torch.tensor([0.0, 1.0], device=device)
        x01 = odeint(vf, x0, ts, method="dopri5", atol=atol, rtol=rtol)
        return x01[-1]


def to_img(x: torch.Tensor) -> torch.Tensor:
    # model space is roughly [-inf, inf]; map to [0,1] for saving
    x = x.clamp(-1, 1)
    x = (x + 1.0) * 0.5
    return x


def main():
    parser = argparse.ArgumentParser(
        description="Sample from trained flow matching model"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./runs/fm_ot_cifar10/ckpt_step200.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./runs/fm_ot_cifar10/samples.png",
        help="Path to save generated samples",
    )
    parser.add_argument(
        "--n", type=int, default=64, help="Number of samples to generate"
    )
    parser.add_argument(
        "--atol", type=float, default=1e-5, help="Absolute tolerance for ODE solver"
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-5, help="Relative tolerance for ODE solver"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Defaults to cuda if available",
    )
    parser.add_argument(
        "--show-trajectory",
        action="store_true",
        help="Show time evolution grid (rows=samples, cols=time)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of time steps to show in trajectory (default: 10)",
    )

    args = parser.parse_args()

    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    net, step = load_ckpt(args.ckpt, device=device)
    print("checkpoint step:", step)

    # Use torchdiffeq dopri5 solver
    if args.show_trajectory:
        # Generate trajectory visualization
        trajectory = dopri5_sample(
            net, n=args.n, device=device, atol=args.atol, rtol=args.rtol, 
            return_trajectory=True, num_steps=args.num_steps
        )
        # trajectory shape: (num_steps, n, 3, 32, 32)
        
        # Rearrange to (n, num_steps, 3, 32, 32) then flatten to (n * num_steps, 3, 32, 32)
        trajectory = trajectory.transpose(0, 1)  # (n, num_steps, 3, 32, 32)
        trajectory_flat = trajectory.reshape(-1, *trajectory.shape[2:])  # (n * num_steps, 3, 32, 32)
        
        # Convert to images
        imgs = to_img(trajectory_flat)
        
        # Make grid: each row is one sample's trajectory through time
        grid = make_grid(imgs, nrow=args.num_steps)
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_image(grid, args.output)
        print(f"saved: {args.output} (trajectory with {args.num_steps} time steps, {args.n} samples)")
    else:
        x1 = dopri5_sample(net, n=args.n, device=device, atol=args.atol, rtol=args.rtol)
        imgs = to_img(x1)
        grid = make_grid(imgs, nrow=8)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_image(grid, args.output)
        print("saved:", args.output)


if __name__ == "__main__":
    main()
