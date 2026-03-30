# sample.py
from __future__ import annotations
import os
import argparse
import torch
from torchvision.utils import save_image, make_grid

from src.models import UNetCIFAR


def load_ckpt(ckpt_path: str, device: str, **model_kwargs):
    """Load checkpoint, returning model with EMA weights if available."""
    ckpt = torch.load(ckpt_path, map_location=device)

    # Recover architecture config saved in checkpoint, fall back to kwargs/defaults
    arch = ckpt.get("arch", {})
    for k, v in model_kwargs.items():
        if v is not None:
            arch[k] = v

    net = UNetCIFAR(**arch).to(device)
    net.load_state_dict(ckpt["model"], strict=True)

    ema_state = ckpt.get("ema", None)
    if ema_state is not None:
        net.load_state_dict(ema_state, strict=True)
        print("Loaded EMA weights for sampling.")
    else:
        print("No EMA in checkpoint; using raw model weights.")

    step = ckpt.get("step", None)
    image_size = ckpt.get("image_size", 32)
    return net, step, image_size


@torch.no_grad()
def dopri5_sample(
    net, n: int = 64, image_size: int = 32, channels: int = 3,
    device: str = "cuda", atol=1e-5, rtol=1e-5,
    return_trajectory=False, num_steps=10,
):
    """ODE sampling using torchdiffeq dopri5 solver."""
    from torchdiffeq import odeint

    net.eval()
    x0 = torch.randn(n, channels, image_size, image_size, device=device)

    class VF(torch.nn.Module):
        def forward(self, t, x):
            t_batch = torch.full(
                (x.shape[0],), float(t), device=x.device, dtype=x.dtype
            )
            return net(x, t_batch)

    vf = VF().to(device)

    if return_trajectory:
        ts = torch.linspace(0.0, 1.0, num_steps, device=device)
        trajectory = odeint(vf, x0, ts, method="dopri5", atol=atol, rtol=rtol)
        return trajectory
    else:
        ts = torch.tensor([0.0, 1.0], device=device)
        x01 = odeint(vf, x0, ts, method="dopri5", atol=atol, rtol=rtol)
        return x01[-1]


def to_img(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-1, 1)
    x = (x + 1.0) * 0.5
    return x


def main():
    parser = argparse.ArgumentParser(
        description="Sample from trained flow matching model"
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--output", type=str, default="./samples.png", help="Output path")
    parser.add_argument("--n", type=int, default=64, help="Number of samples")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--show-trajectory", action="store_true")
    parser.add_argument("--num-steps", type=int, default=10)

    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    net, step, image_size = load_ckpt(args.ckpt, device=device)
    print(f"checkpoint step: {step}, image_size: {image_size}")

    if args.show_trajectory:
        trajectory = dopri5_sample(
            net, n=args.n, image_size=image_size, device=device,
            atol=args.atol, rtol=args.rtol,
            return_trajectory=True, num_steps=args.num_steps,
        )
        trajectory = trajectory.transpose(0, 1)
        trajectory_flat = trajectory.reshape(-1, *trajectory.shape[2:])
        imgs = to_img(trajectory_flat)
        grid = make_grid(imgs, nrow=args.num_steps)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_image(grid, args.output)
        print(f"saved: {args.output} (trajectory with {args.num_steps} time steps, {args.n} samples)")
    else:
        x1 = dopri5_sample(
            net, n=args.n, image_size=image_size, device=device,
            atol=args.atol, rtol=args.rtol,
        )
        imgs = to_img(x1)
        grid = make_grid(imgs, nrow=8)
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_image(grid, args.output)
        print("saved:", args.output)


if __name__ == "__main__":
    main()
