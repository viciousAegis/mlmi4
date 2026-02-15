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
    net: UNetCIFAR, n: int = 64, device: str = "cuda", atol=1e-5, rtol=1e-5
):
    """
    Paper-style ODE solver using torchdiffeq.
    Install: pip install torchdiffeq
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
    ts = torch.tensor([0.0, 1.0], device=device)
    x01 = odeint(vf, x0, ts, method="dopri5", atol=atol, rtol=rtol)
    x1 = x01[-1]
    return x1


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

    args = parser.parse_args()

    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    net, step = load_ckpt(args.ckpt, device=device)
    print("checkpoint step:", step)

    # Use torchdiffeq dopri5 solver
    x = dopri5_sample(net, n=args.n, device=device, atol=args.atol, rtol=args.rtol)

    imgs = to_img(x)
    grid = make_grid(imgs, nrow=8)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_image(grid, args.output)
    print("saved:", args.output)


if __name__ == "__main__":
    main()
