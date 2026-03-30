"""ODE solvers for sampling: Euler, Midpoint, RK4, and adaptive dopri5 with NFE counting."""
from __future__ import annotations
import torch
import torch.nn as nn


class NFECounter(nn.Module):
    """Wraps a vector field module and counts the number of function evaluations."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.nfe = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.nfe += 1
        return self.net(x, t)

    def reset(self):
        self.nfe = 0


def _expand_t(t_scalar: float, batch_size: int, device, dtype):
    return torch.full((batch_size,), t_scalar, device=device, dtype=dtype)


@torch.no_grad()
def euler_sample(net: nn.Module, x0: torch.Tensor, nfe: int) -> torch.Tensor:
    """Euler method with `nfe` fixed steps from t=0 to t=1."""
    x = x0.clone()
    dt = 1.0 / nfe
    for i in range(nfe):
        t = _expand_t(i * dt, x.shape[0], x.device, x.dtype)
        x = x + dt * net(x, t)
    return x


@torch.no_grad()
def midpoint_sample(net: nn.Module, x0: torch.Tensor, nfe: int) -> torch.Tensor:
    """Midpoint method. Uses 2 function evaluations per step, so nfe must be even.
    Total NFE = nfe (each step uses nfe/2 midpoint steps with 2 evals each)."""
    assert nfe % 2 == 0, "Midpoint method requires even NFE (2 evals per step)"
    n_steps = nfe // 2
    dt = 1.0 / n_steps
    x = x0.clone()
    for i in range(n_steps):
        t_i = i * dt
        t_vec = _expand_t(t_i, x.shape[0], x.device, x.dtype)
        k1 = net(x, t_vec)
        t_mid_vec = _expand_t(t_i + 0.5 * dt, x.shape[0], x.device, x.dtype)
        k2 = net(x + 0.5 * dt * k1, t_mid_vec)
        x = x + dt * k2
    return x


@torch.no_grad()
def rk4_sample(net: nn.Module, x0: torch.Tensor, nfe: int) -> torch.Tensor:
    """Classical RK4. Uses 4 function evaluations per step, so nfe must be divisible by 4.
    Total NFE = nfe."""
    assert nfe % 4 == 0, "RK4 requires NFE divisible by 4"
    n_steps = nfe // 4
    dt = 1.0 / n_steps
    x = x0.clone()
    for i in range(n_steps):
        t_i = i * dt
        t1 = _expand_t(t_i, x.shape[0], x.device, x.dtype)
        t2 = _expand_t(t_i + 0.5 * dt, x.shape[0], x.device, x.dtype)
        t3 = _expand_t(t_i + dt, x.shape[0], x.device, x.dtype)

        k1 = net(x, t1)
        k2 = net(x + 0.5 * dt * k1, t2)
        k3 = net(x + 0.5 * dt * k2, t2)
        k4 = net(x + dt * k3, t3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


@torch.no_grad()
def dopri5_sample(net: nn.Module, x0: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-5):
    """Adaptive dopri5 solver via torchdiffeq. Returns (samples, nfe)."""
    from torchdiffeq import odeint

    counter = NFECounter(net)

    class VF(nn.Module):
        def forward(self, t, x):
            t_batch = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
            return counter(x, t_batch)

    vf = VF().to(x0.device)
    ts = torch.tensor([0.0, 1.0], device=x0.device)
    result = odeint(vf, x0, ts, method="dopri5", atol=atol, rtol=rtol)
    return result[-1], counter.nfe


@torch.no_grad()
def dopri5_trajectory(net: nn.Module, x0: torch.Tensor, num_steps: int = 10,
                      atol: float = 1e-5, rtol: float = 1e-5):
    """Adaptive dopri5 returning trajectory at evenly-spaced times. Returns (trajectory, nfe)."""
    from torchdiffeq import odeint

    counter = NFECounter(net)

    class VF(nn.Module):
        def forward(self, t, x):
            t_batch = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
            return counter(x, t_batch)

    vf = VF().to(x0.device)
    ts = torch.linspace(0.0, 1.0, num_steps, device=x0.device)
    trajectory = odeint(vf, x0, ts, method="dopri5", atol=atol, rtol=rtol)
    return trajectory, counter.nfe


SOLVERS = {
    "euler": euler_sample,
    "midpoint": midpoint_sample,
    "rk4": rk4_sample,
}
