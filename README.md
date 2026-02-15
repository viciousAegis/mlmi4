# MLMI4 - Flow Matching on CIFAR-10 and ImageNet

Training script for Flow Matching models with support for both Optimal Transport (OT) and diffusion probability paths on CIFAR-10 and ImageNet datasets.

## Usage

### Installation
This project uses UV to manage dependencies. To run, first install UV and then set up the environment:
```bash
pip install uv
uv sync
```
Everything should be installed in a local `.venv` directory, and work out of the box.

### Basic Training

Run with default settings (CIFAR-10, OT path):
```bash
python training.py
```

### Dataset Selection

Choose between CIFAR-10 or ImageNet:

```bash
# Train on CIFAR-10 (default, 32x32)
python training.py --dataset cifar10

# Train on ImageNet (64x64 by default)
python training.py --dataset imagenet --data-root /path/to/imagenet

# Train on ImageNet with custom image size
python training.py --dataset imagenet --data-root /path/to/imagenet --image-size 128
```

**ImageNet Setup:**
ImageNet requires the standard directory structure:
```
data_root/
  train/
    n01440764/
      *.JPEG
  val/
    n01440764/
      *.JPEG
```

### Path Selection

Choose between Optimal Transport (OT) or diffusion paths:

```bash
# Train with OT path (default)
python training.py --path-type ot

# Train with diffusion path
python training.py --path-type diffusion
```

The path type determines how the model learns to transform noise to data:
- **OT (Optimal Transport)**: Uses a Gaussian probability path with linear interpolation
- **Diffusion**: Uses a VP-diffusion path with cosine noise schedule

### Using Config File

You can use a YAML config file to specify all training parameters:
```bash
python training.py --config config_example.yaml
```

### Overriding Config with Command-Line Arguments

Command-line arguments override config file settings:
```bash
python training.py --config config_example.yaml --batch-size 128 --lr 0.001
```

## Weights & Biases Integration

### Enable W&B Logging

```bash
# First, login to wandb
wandb login

# Enable wandb logging
python training.py --use-wandb --wandb-project my-project

# With config file
python training.py --config config_example.yaml --use-wandb
```

### W&B Features

The integration logs:
- Training loss and learning rate
- Validation loss
- Training speed (steps/second)
- Model checkpoints as artifacts
- All hyperparameters in the config

### W&B Options

- `--use-wandb` / `--no-wandb`: Enable/disable W&B logging
- `--wandb-project`: Project name (default: "mlmi4-cifar10")
- `--wandb-entity`: Team/entity name (optional)
- `--wandb-name`: Custom run name (optional)
- `--wandb-tags`: Space-separated tags (e.g., `--wandb-tags baseline v1`)

### Available Command-Line Arguments

- `--config`: Path to YAML config file
- `--force-cpu`: Force CPU usage
- `--dataset`: Dataset - "cifar10" or "imagenet" (default: "cifar10")
- `--data-root`: Data directory path (default: "./data")
- `--image-size`: Image size for ImageNet (default: 64)
- `--batch-size`: Per-step batch size
- `--effective-batch`: Target effective batch size
- `--val-size`: Validation set size
- `--num-workers`: Number of data loading workers
- `--total-steps`: Total training steps
- `--log-every`: Log every N steps
- `--val-every`: Validate every N steps
- `--ckpt-every`: Save checkpoint every N steps
- `--lr`: Learning rate
- `--warmup-steps`: Warmup steps
- `--poly-power`: Polynomial decay power
- `--weight-decay`: Weight decay
- `--path-type`: Path type - "ot" or "diffusion" (default: "ot")
- `--sigma-min`: Minimum sigma for OT path (default: 0.01)
- `--diffusion-s`: Schedule parameter for diffusion path (default: 0.008)
- `--base-ch`: Base channel size
- `--num-res-blocks`: Number of residual blocks
- `--num-heads`: Number of attention heads
- `--dropout`: Dropout rate
- `--use-amp` / `--no-amp`: Enable/disable automatic mixed precision
- `--ema` / `--no-ema`: Enable/disable EMA
- `--ema-decay`: EMA decay rate
- `--out-dir`: Output directory

## Config File Format

See [config_example.yaml](config_example.yaml) for an example configuration file.
