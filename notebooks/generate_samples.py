"""
Run this in Google Colab or any GPU environment.
Generates:
1. Two denoising trajectory grids (2 rows x 10 cols each)
2. Two final sample grids (4 rows x 8 cols = 32 samples each)

Setup cell (run first):
    !pip install torch torchvision torchdiffeq huggingface_hub
    !git clone https://github.com/viciousAegis/mlmi4.git
    %cd mlmi4
"""
import os
import torch
from torchvision.utils import save_image, make_grid
from huggingface_hub import hf_hub_download

# --- Config ---
REPO_ID = "shanai13/MLMI4_flow_OT"
CKPT_NAME = "ckpt_step125000.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_TIME_STEPS = 10
SEED = 42

# --- Download checkpoint ---
print("Downloading checkpoint...")
ckpt_path = hf_hub_download(repo_id=REPO_ID, filename=CKPT_NAME)

# --- Load model ---
from src.models import UNetCIFAR

ckpt = torch.load(ckpt_path, map_location=DEVICE)
arch = ckpt.get("arch", {})
image_size = ckpt.get("image_size", 32)

net = UNetCIFAR(**arch).to(DEVICE)
ema_state = ckpt.get("ema", None)
if ema_state is not None:
    net.load_state_dict(ema_state, strict=True)
    print("Loaded EMA weights.")
else:
    net.load_state_dict(ckpt["model"], strict=True)
net.eval()
print(f"Model loaded. Image size: {image_size}x{image_size}")

# --- Helper ---
def to_img(x):
    return (x.clamp(-1, 1) + 1) * 0.5

# --- ODE setup ---
from torchdiffeq import odeint

class VF(torch.nn.Module):
    def forward(self, t, x):
        t_batch = torch.full((x.shape[0],), float(t), device=x.device, dtype=x.dtype)
        return net(x, t_batch)

vf = VF().to(DEVICE)
os.makedirs("outputs", exist_ok=True)

# --- 1. Two denoising trajectory grids (2 rows x 10 cols each) ---
ts = torch.linspace(0.0, 1.0, N_TIME_STEPS, device=DEVICE)

for i in range(2):
    print(f"\nGenerating denoising trajectory set {i+1}...")
    torch.manual_seed(SEED + i * 100)
    x0 = torch.randn(2, 3, image_size, image_size, device=DEVICE)

    with torch.no_grad():
        trajectory = odeint(vf, x0, ts, method="dopri5", atol=1e-5, rtol=1e-5)
        # (N_TIME_STEPS, 2, 3, H, W)

    trajectory = trajectory.transpose(0, 1)  # (2, N_TIME_STEPS, 3, H, W)
    traj_flat = trajectory.reshape(-1, 3, image_size, image_size)
    traj_imgs = to_img(traj_flat)
    traj_grid = make_grid(traj_imgs, nrow=N_TIME_STEPS, padding=1, pad_value=1)
    save_image(traj_grid, f"outputs/denoising_trajectory_{i+1}.png")
    print(f"Saved: outputs/denoising_trajectory_{i+1}.png")

# --- 2. Two final sample grids (4 rows x 8 cols = 32 each) ---
ts_final = torch.tensor([0.0, 1.0], device=DEVICE)

for i in range(2):
    print(f"\nGenerating sample grid {i+1} (32 samples)...")
    torch.manual_seed(SEED + 200 + i * 100)
    x0 = torch.randn(32, 3, image_size, image_size, device=DEVICE)

    with torch.no_grad():
        result = odeint(vf, x0, ts_final, method="dopri5", atol=1e-5, rtol=1e-5)
        samples = result[-1]

    sample_imgs = to_img(samples)
    sample_grid = make_grid(sample_imgs, nrow=8, padding=1, pad_value=1)
    save_image(sample_grid, f"outputs/samples_grid_{i+1}.png")
    print(f"Saved: outputs/samples_grid_{i+1}.png")

print("\nDone! Check the outputs/ directory.")
