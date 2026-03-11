from __future__ import annotations

import torch


def kappa_t(t: torch.Tensor, schedule: str = "cubic") -> torch.Tensor:
    if schedule == "linear":
        return t
    if schedule == "square":
        return t * t
    if schedule == "cubic":
        return t * t * t
    raise ValueError(f"Unknown schedule: {schedule}")


def discrete_corrupt_and_target(
    x1: torch.Tensor,
    mask_token_id: int,
    schedule: str = "cubic",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Corrupt tokens with a time-dependent mask ratio for discrete FM-style training."""
    if x1.ndim != 2:
        raise ValueError("x1 must have shape (B, L)")

    bsz = x1.shape[0]
    device = x1.device
    t = torch.rand(bsz, device=device)
    kappa = kappa_t(t, schedule=schedule).clamp(0.0, 1.0)
    keep_prob = kappa[:, None]

    keep = torch.rand_like(x1, dtype=torch.float32) < keep_prob
    x_t = torch.where(keep, x1, torch.full_like(x1, mask_token_id))
    target_mask = ~keep
    return t, x_t, target_mask


def reveal_probability(
    t_cur: torch.Tensor,
    t_next: torch.Tensor,
    schedule: str = "cubic",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Probability to reveal currently masked tokens between two times."""
    m_cur = (1.0 - kappa_t(t_cur, schedule=schedule)).clamp_min(eps)
    m_next = (1.0 - kappa_t(t_next, schedule=schedule)).clamp_min(0.0)
    r = (m_cur - m_next) / m_cur
    return r.clamp(0.0, 1.0)
