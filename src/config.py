"""Configuration and argument parsing for training."""
from __future__ import annotations
import argparse
import yaml
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # device
    force_cpu: bool = False

    # data
    dataset: str = "cifar10"           # "cifar10" or "imagenet"
    data_root: str = "./data"          # data directory
    image_size: int = 64               # image size (for ImageNet)
    batch_size: int = 64               # per-step batch
    effective_batch: int = 256         # target effective batch
    val_size: int = 5000
    num_workers: int = 4

    # training length
    total_steps: int = 50_000          # set to 391_000 for paper-like
    log_every: int = 100
    val_every: int = 2_000
    ckpt_every: int = 5_000

    # optimization
    lr: float = 5e-4
    warmup_steps: int = 45_000
    poly_power: float = 2.0
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0

    # FM / OT path
    path_type: str = "ot"            # "ot" or "diffusion"
    sigma_min: float = 0.01            # for OT path
    beta_min: float = 0.1             # for VP diffusion path
    beta_max: float = 20.0            # for VP diffusion path
    eps_t: float = 1e-5               # time clipping for diffusion path

    # model
    base_ch: int = 256
    channel_mults: tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int, ...] = (16,)
    num_heads: int = 4
    dropout: float = 0.0

    # AMP / EMA
    use_amp: bool = True
    ema: bool = True
    ema_decay: float = 0.9999

    # wandb
    use_wandb: bool = False
    wandb_project: str = "mlmi4-cifar10"
    wandb_entity: str | None = None
    wandb_name: str | None = None
    wandb_tags: tuple[str, ...] | None = None

    # misc
    out_dir: str = "./runs/fm_ot_cifar10"


def load_config(config_path: str | None, args: argparse.Namespace) -> TrainConfig:
    """Load config from YAML file and override with command-line arguments."""
    cfg = TrainConfig()
    
    # Load from YAML if provided
    if config_path is not None:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update cfg with values from YAML
        for key, value in config_dict.items():
            if hasattr(cfg, key):
                # Handle tuple conversion for tuples stored as lists in YAML
                field_type = type(getattr(cfg, key))
                if field_type == tuple and isinstance(value, list):
                    value = tuple(value)
                setattr(cfg, key, value)
            else:
                print(f"[warn] Unknown config key in YAML: {key}")
    
    # Override with command-line arguments (if provided)
    for key, value in vars(args).items():
        if key in ['config'] or value is None:
            continue
        if hasattr(cfg, key):
            # Special handling for wandb_tags
            if key == 'wandb_tags' and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
    
    return cfg


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train Flow Matching model on CIFAR-10')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    
    # Device
    parser.add_argument('--force-cpu', action='store_true', default=None,
                        help='Force CPU usage')
    
    # Data
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['cifar10', 'imagenet'],
                        help='Dataset: "cifar10" or "imagenet"')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Data directory path')
    parser.add_argument('--image-size', type=int, default=None,
                        help='Image size for ImageNet (default: 64)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Per-step batch size')
    parser.add_argument('--effective-batch', type=int, default=None,
                        help='Target effective batch size')
    parser.add_argument('--val-size', type=int, default=None,
                        help='Validation set size')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of data loading workers')
    
    # Training length
    parser.add_argument('--total-steps', type=int, default=None,
                        help='Total training steps')
    parser.add_argument('--log-every', type=int, default=None,
                        help='Log every N steps')
    parser.add_argument('--val-every', type=int, default=None,
                        help='Validate every N steps')
    parser.add_argument('--ckpt-every', type=int, default=None,
                        help='Save checkpoint every N steps')
    
    # Optimization
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=None,
                        help='Warmup steps')
    parser.add_argument('--poly-power', type=float, default=None,
                        help='Polynomial decay power')
    parser.add_argument('--weight-decay', type=float, default=None,
                        help='Weight decay')
    
    # FM / OT path
    parser.add_argument('--path-type', type=str, default=None,
                        choices=['ot', 'diffusion'],
                        help='Path type: "ot" or "diffusion"')
    parser.add_argument('--sigma-min', type=float, default=None,
                        help='Minimum sigma for OT path')
    parser.add_argument('--beta-min', type=float, default=None,
                        help='VP diffusion beta_min (default: 0.1)')
    parser.add_argument('--beta-max', type=float, default=None,
                        help='VP diffusion beta_max (default: 20.0)')
    parser.add_argument('--eps-t', type=float, default=None,
                        help='Time clipping for diffusion path (default: 1e-5)')
    
    # Model
    parser.add_argument('--base-ch', type=int, default=None,
                        help='Base channel size')
    parser.add_argument('--num-res-blocks', type=int, default=None,
                        help='Number of residual blocks')
    parser.add_argument('--num-heads', type=int, default=None,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate')
    
    # AMP / EMA
    parser.add_argument('--use-amp', action='store_true', default=None,
                        help='Use automatic mixed precision')
    parser.add_argument('--no-amp', dest='use_amp', action='store_false',
                        help='Disable automatic mixed precision')
    parser.add_argument('--ema', action='store_true', default=None,
                        help='Use EMA')
    parser.add_argument('--no-ema', dest='ema', action='store_false',
                        help='Disable EMA')
    parser.add_argument('--ema-decay', type=float, default=None,
                        help='EMA decay rate')
    
    # Wandb
    parser.add_argument('--use-wandb', action='store_true', default=None,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--no-wandb', dest='use_wandb', action='store_false',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default=None,
                        help='W&B project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='W&B entity/team name')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='W&B run name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=None,
                        help='W&B run tags')
    
    # Misc
    parser.add_argument('--out-dir', type=str, default=None,
                        help='Output directory')
    
    return parser.parse_args()
