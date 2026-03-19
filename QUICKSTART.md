# Quick Start Guide

## Setup

### 1. Clone the repo

```bash
cd /home/<YOUR_CRSid>/rds/hpc-work
git clone git@github.com:viciousAegis/mlmi4.git
cd mlmi4
```

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### 3. Install dependencies

```bash
uv sync
```

### 4. Download CIFAR-10 (on login node, before submitting jobs)

```bash
uv run python3 -c "
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torchvision.datasets import CIFAR10
CIFAR10(root='./data', train=True, download=True)
CIFAR10(root='./data', train=False, download=True)
"
```

### 5. Pre-download Inception weights for FID evaluation

```bash
uv run python3 -c "
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from pytorch_fid.inception import InceptionV3
InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]])
print('Inception weights cached.')
"
```

### 6. Download the trained OT checkpoint

```bash
mkdir -p runs/cifar10_ot
uv run python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='shanai13/MLMI4_flow_OT', filename='ckpt_step125000.pt', local_dir='runs/cifar10_ot')
print('Downloaded.')
"
```

---

## Further Instructions for James: Train FM w/ Diffusion

### Update account name in slurm script

In slurm/run_t3.slurm, update line 6 to include your own account name (e.g. ab1234):

```bash
#SBATCH -A MLMI-<YOUR_ACCOUNT_NAME>-SL2-GPU
```

This will allow you to successfully submit the slurm job in the next step.

### Submit training job

```bash
mkdir -p logs
sbatch slurm/run_t3.slurm
```

This trains the FM w/ Diffusion model on CIFAR-10 for ~195k steps (~10-12 hours on an A100).

**What to expect:**

- Checkpoints saved every 25k steps in `runs/cifar10_diffusion/`
- FID evaluated every 25k steps, logged to WandB (project: `mlmi4-flow-matching`)
- Training loss should settle around ~0.15-0.17
- Best FID should be comparable to or slightly worse than OT (paper: FID=8.06 for FM-Diffusion vs 6.35 for FM-OT)

**When training finishes:**

- Check WandB for the best FID and note which step it occurred at
- Edit `slurm/run_eval_diff.slurm` — update the `CKPT=` line to point to the best checkpoint
- Submit evaluations:

```bash
sbatch slurm/run_eval_diff.slurm
```

### Monitor

```bash
squeue -u $USER
tail -f logs/t3_*.out
```

Ideally also upload this checkpoint to HF so we can include a link in the report.

---

## Evaluate FM w/ OT

Not all of these will we used for the poster, but they will all be in the report.

### Submit evaluation job

```bash
mkdir -p logs
sbatch slurm/run_eval_ot.slurm
```

This runs (~6-10 hours):

- **E1:** FID with dopri5 solver (50k samples) — Table 1
- **E4:** ODE integration error vs NFE — Figure 7 (left)
- **E5:** FID vs NFE with Euler/Midpoint/RK4 — Figure 7 (right)
- **E6:** Sample grid (64 samples) — qualitative results
- **E7:** Trajectory visualisation (noise → image) — Figure 6

**Outputs** will be in `runs/eval_ot/`:

- `fid_dopri5/` — generated samples
- `ode_error_vs_nfe.png` — error plot
- `ode_error_results.json` — raw error data
- `fid_vs_nfe.png` — FID plot
- `fid_vs_nfe_results.json` — raw FID data
- `samples.png` — sample grid
- `trajectory.png` — trajectory visualisation

### Monitor

```bash
squeue -u $USER
tail -f logs/eval_ot_*.out
```

---

## After Both Are Done

We should have:

- FM w/ OT: FID, FID-vs-NFE, ODE-error-vs-NFE, samples, trajectories
- FM w/ Diffusion: FID, FID-vs-NFE, ODE-error-vs-NFE, samples, trajectories

Download results to your local machine:

```bash
scp -r <CRSid>@login.hpc.cam.ac.uk:<path_to_mlmi4>/runs/eval_ot ./results_ot
scp -r <CRSid>@login.hpc.cam.ac.uk:<path_to_mlmi4>/runs/eval_diff ./results_diff
```

---

## Key Expected Results (Paper Table 1, CIFAR-10)

| Model           | NLL (BPD) | FID  | NFE |
| --------------- | --------- | ---- | --- |
| FM w/ Diffusion | 3.10      | 8.06 | 183 |
| FM w/ OT        | 2.99      | 6.35 | 142 |

Our OT model achieved FID=5.95 during training — better than the paper.
