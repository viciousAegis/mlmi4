import torch


@torch.no_grad()
def ot_path_and_target(x1: torch.Tensor, sigma_min: float = 0.01):
    """
    OT-style Gaussian probability path used for Flow Matching training.

    Args:
        x1: data batch in [-1,1], shape (B,C,H,W)
        sigma_min: small >0, endpoint noise scale at t=1

    Returns:
        t: (B,) in [0,1]
        x_t: (B,C,H,W) sampled point on the path
        u_t: (B,C,H,W) target conditional vector field u_t(x_t | x1)
    """

    B, C, H, W = x1.shape
    device = x1.device
    dtype = x1.dtype

    # sample time and noise
    t = torch.rand(B, device=device, dtype=dtype)  # (B,)
    eps = torch.randn_like(x1)  # (B,C,H,W)

    # broadcast time
    t_ = t[:, None, None, None]  # (B,1,1,1)

    # path parameters
    mu_t = t_ * x1  # (B,C,H,W)
    sigma_t = (1.0 - t_) + t_ * sigma_min  # (B,1,1,1) broadcastable

    # sample x_t
    x_t = mu_t + sigma_t * eps

    # target vector field:
    # u_t = dmu + (dsigma/sigma_t) * (x_t - mu_t)
    dmu = x1
    dsigma = sigma_min - 1.0
    u_t = dmu + (dsigma / sigma_t) * (x_t - mu_t)

    return t, x_t, u_t


def alpha_vp(t: torch.Tensor, beta_min: float = 0.1, beta_max: float = 20.0) -> torch.Tensor:
    """
    VP linear-beta schedule alpha_bar(t) as used in the paper (Appendix E.1).
    beta(s) = beta_min + (beta_max - beta_min) * s
    alpha_bar(t) = exp(-0.5 * integral_0^t beta(s) ds)
                 = exp(-0.5 * (beta_min * t + 0.5 * (beta_max - beta_min) * t^2))
    """
    log_abar = -0.5 * (beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2)
    return torch.exp(log_abar)


def diffusion_path_and_target(
    x1: torch.Tensor, beta_min: float = 0.1, beta_max: float = 20.0, eps_t: float = 1e-5
):
    """
    VP-diffusion probability path (paper Eq. 18, Appendix E.1):
      p_t(x|x1) = N(x | alpha_{1-t} * x1, (1 - alpha_{1-t}^2) * I)

    Uses time-reversed alpha_bar: alpha_bar is evaluated at (1-t) so that
    t=0 corresponds to noise and t=1 corresponds to data.

    Time is sampled from [eps_t, 1-eps_t] to avoid numerical issues.

    Returns:
      t: (B,)
      x_t: (B,C,H,W)
      u_t: (B,C,H,W) target conditional vector field
    """
    assert x1.ndim == 4
    B = x1.shape[0]
    device = x1.device
    dtype = x1.dtype

    # sample t in [eps_t, 1-eps_t] and eps
    t = torch.rand(B, device=device, dtype=dtype) * (1.0 - 2 * eps_t) + eps_t
    eps = torch.randn_like(x1)

    # Use autograd to compute d/dt of mu_scale and sig
    # enable_grad needed so this works even inside @torch.no_grad() contexts
    with torch.enable_grad():
        t_req = t.detach().clone().requires_grad_(True)

        # Time-reversed: evaluate alpha at (1-t) per paper Eq. 18
        # alpha_vp returns alpha_t = exp(-0.5 * integral), NOT alpha_bar
        # Paper: mu_t = alpha_{1-t} * x1, sigma_t = sqrt(1 - alpha_{1-t}^2)
        alpha = alpha_vp(1.0 - t_req, beta_min=beta_min, beta_max=beta_max)
        mu_scale = alpha
        sig = torch.sqrt(1.0 - alpha ** 2)

        # Compute d(mu_scale)/dt and d(sig)/dt via autograd
        dmu_scale = torch.autograd.grad(
            mu_scale.sum(), t_req, create_graph=False, retain_graph=True
        )[0]
        dsig = torch.autograd.grad(sig.sum(), t_req, create_graph=False)[0]

    # broadcast
    mu_scale_ = mu_scale.detach()[:, None, None, None]
    sig_ = sig.detach()[:, None, None, None]

    mu_t = mu_scale_ * x1
    x_t = mu_t + sig_ * eps

    dmu = dmu_scale.detach()[:, None, None, None] * x1
    dsig_ = dsig.detach()[:, None, None, None]

    # u_t = dmu + (dsig/sig) * (x_t - mu_t)
    u_t = dmu + (dsig_ / sig_) * (x_t - mu_t)

    return t.detach(), x_t.detach(), u_t.detach()


def get_path_and_target(
    x1: torch.Tensor, path_type: str = "ot", sigma_min: float = 0.01,
    beta_min: float = 0.1, beta_max: float = 20.0, eps_t: float = 1e-5,
):
    """
    Unified interface for path selection.

    Args:
        x1: data batch
        path_type: "ot" or "diffusion"
        sigma_min: for OT path
        beta_min: for VP diffusion path
        beta_max: for VP diffusion path
        eps_t: time clipping for diffusion path

    Returns:
        t, x_t, u_t
    """
    if path_type == "ot":
        return ot_path_and_target(x1, sigma_min=sigma_min)
    elif path_type == "diffusion":
        return diffusion_path_and_target(x1, beta_min=beta_min, beta_max=beta_max, eps_t=eps_t)
    else:
        raise ValueError(f"Unknown path_type: {path_type}. Use 'ot' or 'diffusion'.")


def main():
    # quick sanity test on fake data
    x1 = torch.empty(8, 3, 32, 32).uniform_(-1, 1)
    t, x_t, u_t = ot_path_and_target(x1, sigma_min=0.01)

    print("t:", t.shape, t.min().item(), t.max().item())
    print("x_t:", x_t.shape, x_t.dtype, float(x_t.min()), float(x_t.max()))
    print("u_t:", u_t.shape, u_t.dtype, float(u_t.min()), float(u_t.max()))

    # sanity: when t is very close to 1, x_t ~ x1 (small noise)
    t2 = torch.ones(8) * 0.999
    t2_ = t2[:, None, None, None]
    eps = torch.randn_like(x1)
    sigma_min = 0.01
    mu_t2 = t2_ * x1
    sigma_t2 = (1 - t2_) + t2_ * sigma_min
    x_t2 = mu_t2 + sigma_t2 * eps
    print("mean |x_t2 - x1| (t≈1):", (x_t2 - x1).abs().mean().item())

    # check diffusion path
    t3, x_t3, u_t3 = diffusion_path_and_target(x1)
    print("diffusion t:", t3.shape, t3.min().item(), t3.max().item())
    print(
        "diffusion x_t:", x_t3.shape, x_t3.dtype, float(x_t3.min()), float(x_t3.max())
    )
    print(
        "diffusion u_t:", u_t3.shape, u_t3.dtype, float(u_t3.min()), float(u_t3.max())
    )


if __name__ == "__main__":
    main()
