# train.py
from __future__ import annotations
import os
import time
from dataclasses import asdict

import torch
import wandb

from src.datasets import get_cifar10_loaders
from src.sample import sample_with_ot
from src.models import UNetCIFAR
from src.config import TrainConfig, parse_args, load_config
from src.utils import EMA, pick_device, lr_at_step, evaluate_loss, save_ckpt


def main():
    args = parse_args()
    cfg = load_config(args.config, args)

    device = pick_device(cfg.force_cpu)
    os.makedirs(cfg.out_dir, exist_ok=True)
    print("device:", device)
    print("out_dir:", cfg.out_dir)

    # Initialize wandb
    if cfg.use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_name,
            tags=list(cfg.wandb_tags) if cfg.wandb_tags else None,
            config=asdict(cfg),
        )
        # Update cfg with any wandb sweep overrides
        if wandb.config:
            for key, value in wandb.config.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)

    # CPU-friendly defaults
    if device == "cpu":
        cfg.batch_size = 8
        cfg.effective_batch = 32
        cfg.num_workers = 0
        cfg.total_steps = 200
        cfg.warmup_steps = 50
        cfg.log_every = 20
        cfg.val_every = 100
        cfg.ckpt_every = 200
        cfg.use_amp = False
        cfg.ema = False

    accum_steps = max(1, cfg.effective_batch // cfg.batch_size)
    if cfg.effective_batch % cfg.batch_size != 0:
        print(f"[warn] effective_batch not divisible by batch_size; using accum_steps={accum_steps}")

    loaders = get_cifar10_loaders(
        batch_size=cfg.batch_size,
        val_size=cfg.val_size,
        num_workers=cfg.num_workers,
    )

    net = UNetCIFAR(
        base_ch=cfg.base_ch,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)

    use_amp = cfg.use_amp and (device == "cuda")
    autocast = torch.amp.autocast
    scaler = torch.amp.GradScaler(enabled=use_amp)

    ema = EMA(net, cfg.ema_decay) if cfg.ema else None

    train_it = iter(loaders.train)
    t0 = time.time()

    net.train()
    for step in range(1, cfg.total_steps + 1):
        # set LR
        lr = lr_at_step(cfg, step)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(accum_steps):
            try:
                x1, _ = next(train_it)
            except StopIteration:
                train_it = iter(loaders.train)
                x1, _ = next(train_it)

            x1 = x1.to(device, non_blocking=(device == "cuda"))
            t, x_t, u_t = sample_with_ot(x1, sigma_min=cfg.sigma_min)

            if use_amp:
                with autocast("cuda", dtype=torch.float16):
                    v = net(x_t, t)
                    loss = (v - u_t).pow(2).mean() / accum_steps
                scaler.scale(loss).backward()
            else:
                v = net(x_t, t)
                loss = (v - u_t).pow(2).mean() / accum_steps
                loss.backward()

            total_loss += float(loss.item())

        if use_amp:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        if ema is not None:
            ema.update(net)

        # logging
        if step % cfg.log_every == 0 or step == 1:
            dt = time.time() - t0
            steps_per_s = step / max(1e-6, dt)
            print(f"step {step:06d} | lr {lr:.3e} | train_loss {total_loss:.6f} | {steps_per_s:.2f} steps/s")
            
            if cfg.use_wandb:
                wandb.log({
                    "train/loss": total_loss,
                    "train/lr": lr,
                    "train/steps_per_s": steps_per_s,
                    "step": step,
                })

        # validation
        if step % cfg.val_every == 0:
            val_loss = evaluate_loss(net, loaders.val, device, cfg.sigma_min, max_batches=20)
            print(f"           | val_loss {val_loss:.6f}")
            
            if cfg.use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "step": step,
                })

        # checkpoint
        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_step{step}.pt")
            save_ckpt(ckpt_path, net, opt, step, ema)
            print(f"saved: {ckpt_path}")
            
            # # Save checkpoint to wandb as artifact
            # if cfg.use_wandb:
            #     artifact = wandb.Artifact(f"model-step{step}", type="model")
            #     artifact.add_file(ckpt_path)
            #     wandb.log_artifact(artifact)
    
    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
