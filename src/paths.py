import torch

@torch.no_grad()
def sample_with_ot(x1: torch.Tensor, sigma_min: float = 0.01):
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
    t = torch.rand(B, device=device, dtype=dtype)                 # (B,)
    eps = torch.randn_like(x1)                                    # (B,C,H,W)

    # broadcast time
    t_ = t[:, None, None, None]                                   # (B,1,1,1)

    # path parameters
    mu_t = t_ * x1                                                # (B,C,H,W)
    sigma_t = (1.0 - t_) + t_ * sigma_min                         # (B,1,1,1) broadcastable

    # sample x_t
    x_t = mu_t + sigma_t * eps

    # target vector field:
    # u_t = dmu + (dsigma/sigma_t) * (x_t - mu_t)
    dmu = x1
    dsigma = (sigma_min - 1.0)
    u_t = dmu + (dsigma / sigma_t) * (x_t - mu_t)

    return t, x_t, u_t

def main():
    # quick sanity test on fake data
    x1 = torch.empty(8, 3, 32, 32).uniform_(-1, 1)
    t, x_t, u_t = sample_with_ot(x1, sigma_min=0.01)

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

if __name__ == "__main__":
    main()