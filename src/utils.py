"""Training utilities including EMA, learning rate scheduling, and evaluation."""
from __future__ import annotations
import torch
import torch.nn as nn

from src.sample import sample_with_ot
from src.config import TrainConfig


class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in msd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


def pick_device(force_cpu: bool) -> str:
    """Select device for training."""
    if force_cpu:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def lr_at_step(cfg: TrainConfig, step: int) -> float:
    """Calculate learning rate with warmup and polynomial decay."""
    # warmup then polynomial decay to 0
    if step <= cfg.warmup_steps:
        return cfg.lr * step / max(1, cfg.warmup_steps)
    # after warmup, decay over remaining steps
    remain = max(1, cfg.total_steps - cfg.warmup_steps)
    s = min(cfg.total_steps - cfg.warmup_steps, step - cfg.warmup_steps)
    frac = 1.0 - (s / remain)
    return cfg.lr * (frac ** cfg.poly_power)


@torch.no_grad()
def evaluate_loss(net: nn.Module, loader, device: str, sigma_min: float, max_batches: int = 20) -> float:
    """Evaluate model loss on validation set."""
    net.eval()
    losses = []
    it = iter(loader)
    for _ in range(max_batches):
        try:
            x1, _ = next(it)
        except StopIteration:
            break
        x1 = x1.to(device)
        t, x_t, u_t = sample_with_ot(x1, sigma_min=sigma_min)
        v = net(x_t, t)
        loss = (v - u_t).pow(2).mean()
        losses.append(loss.item())
    net.train()
    return float(sum(losses) / max(1, len(losses)))


def save_ckpt(path: str, net: nn.Module, opt: torch.optim.Optimizer, step: int, ema: EMA | None):
    """Save training checkpoint."""
    payload = {
        "step": step,
        "model": net.state_dict(),
        "opt": opt.state_dict(),
        "ema": (ema.shadow if ema is not None else None),
    }
    torch.save(payload, path)
