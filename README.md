# MLMI4 - Flow Matching on CIFAR-10

Training script for Flow Matching models on CIFAR-10 dataset.

## Usage

### Basic Training

Run with default settings:
```bash
python training.py
```

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
- `--sigma-min`: Minimum sigma for noise schedule
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
