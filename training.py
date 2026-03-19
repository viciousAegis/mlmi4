# train.py
from __future__ import annotations
import os
import time
import copy
import tempfile
from dataclasses import asdict

import torch
import wandb

from src.datasets import get_dataloaders
from src.paths import get_path_and_target
from src.models import UNetCIFAR
from src.config import TrainConfig, parse_args, load_config
from src.utils import EMA, pick_device, lr_at_step, evaluate_loss, save_ckpt


@torch.no_grad()
def evaluate_training_fid(net, ema, cfg, loaders, device, arch_kwargs):
    """Compute FID during training using current (EMA) model."""
    from evaluate_fid import generate_samples, save_images_to_dir, compute_fid

    # Use EMA weights if available
    eval_net = copy.deepcopy(net)
    if ema is not None:
        ema.copy_to(eval_net)
    eval_net.eval()

    images, avg_nfe = generate_samples(
        eval_net, n=cfg.fid_n_samples, image_size=loaders.image_size,
        channels=3, device=device, solver="dopri5", nfe=None,
        atol=1e-5, rtol=1e-5, batch_size=min(256, cfg.fid_n_samples),
    )
    sample_dir = tempfile.mkdtemp(prefix="fm_train_fid_")
    save_images_to_dir(images, sample_dir)
    fid = compute_fid(sample_dir, cfg.dataset, cfg.data_root, loaders.image_size, device)

    # Clean up
    import shutil
    shutil.rmtree(sample_dir, ignore_errors=True)
    return fid, avg_nfe


@torch.no_grad()
def evaluate_training_nfe(net, ema, cfg, loaders, device):
    """Measure average adaptive solver NFE."""
    from src.solvers import dopri5_sample as dopri5_with_nfe

    eval_net = copy.deepcopy(net)
    if ema is not None:
        ema.copy_to(eval_net)
    eval_net.eval()

    x0 = torch.randn(cfg.nfe_n_samples, 3, loaders.image_size, loaders.image_size, device=device)
    _, nfe = dopri5_with_nfe(eval_net, x0, atol=1e-5, rtol=1e-5)
    return nfe


def load_resume_ckpt(cfg, net, opt, ema, device):
    """Resume training from the latest checkpoint in out_dir if --resume is set."""
    import glob
    ckpts = glob.glob(os.path.join(cfg.out_dir, "ckpt_step*.pt"))
    if not ckpts:
        print("No checkpoint found to resume from. Starting from scratch.")
        return 0

    max_ckpt = max(ckpts, key=lambda x: int(x.split("step")[1].split(".")[0])) # extract step numbers from ckpt paths and select ckpt with max steps
    ckpt_path = max_ckpt
    print(f"Resuming from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    net.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.shadow = ckpt["ema"]

    start_step = ckpt["step"]
    print(f"Resumed at step {start_step}")
    return start_step


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
            resume="allow",
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
        print(
            f"[warn] effective_batch not divisible by batch_size; using accum_steps={accum_steps}"
        )

    loaders = get_dataloaders(
        dataset=cfg.dataset,
        root=cfg.data_root,
        batch_size=cfg.batch_size,
        image_size=cfg.image_size,
        val_size=cfg.val_size,
        num_workers=cfg.num_workers,
    )

    print(f"Dataset: {cfg.dataset}")
    print(
        f"Image size: {loaders.image_size}x{loaders.image_size}, Channels: {loaders.num_channels}"
    )

    arch_kwargs = dict(
        base_ch=cfg.base_ch,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        attn_resolutions=cfg.attn_resolutions,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    )
    net = UNetCIFAR(**arch_kwargs).to(device)

    opt = torch.optim.Adam(
        net.parameters(), lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay
    )

    use_amp = cfg.use_amp and (device == "cuda")
    autocast = torch.amp.autocast
    scaler = torch.amp.GradScaler(enabled=use_amp)

    ema = EMA(net, cfg.ema_decay) if cfg.ema else None

    # Resume from checkpoint if requested
    start_step = 0
    if cfg.resume:
        start_step = load_resume_ckpt(cfg, net, opt, ema, device)

    train_it = iter(loaders.train)
    t0 = time.time()

    net.train()
    for step in range(start_step + 1, cfg.total_steps + 1):
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
            t, x_t, u_t = get_path_and_target(
                x1, path_type=cfg.path_type, sigma_min=cfg.sigma_min,
                beta_min=cfg.beta_min, beta_max=cfg.beta_max, eps_t=cfg.eps_t,
            )

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
            elapsed_steps = step - start_step
            steps_per_s = elapsed_steps / max(1e-6, dt)
            print(
                f"step {step:06d} | lr {lr:.3e} | train_loss {total_loss:.6f} | {steps_per_s:.2f} steps/s"
            )

            if cfg.use_wandb:
                wandb.log(
                    {
                        "train/loss": total_loss,
                        "train/lr": lr,
                        "train/steps_per_s": steps_per_s,
                        "step": step,
                    }
                )

        # validation
        if step % cfg.val_every == 0:
            val_loss = evaluate_loss(net, loaders.val, device, cfg, max_batches=20)
            print(f"           | val_loss {val_loss:.6f}")

            if cfg.use_wandb:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "step": step,
                    }
                )

        # checkpoint FIRST (before evals that might fail)
        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_step{step}.pt")
            save_ckpt(ckpt_path, net, opt, step, ema,
                      arch=arch_kwargs, image_size=loaders.image_size)
            print(f"saved: {ckpt_path}")

        # FID evaluation (Figure 5) — after checkpoint so failures don't lose progress
        if cfg.fid_every > 0 and step % cfg.fid_every == 0:
            try:
                fid, fid_nfe = evaluate_training_fid(net, ema, cfg, loaders, device, arch_kwargs)
                print(f"           | FID {fid:.2f} (NFE={fid_nfe:.0f})")
                if cfg.use_wandb:
                    wandb.log({"eval/fid": fid, "eval/fid_nfe": fid_nfe, "step": step})
            except Exception as e:
                print(f"           | FID evaluation failed: {e}")

        # NFE tracking (Figure 10)
        if cfg.nfe_every > 0 and step % cfg.nfe_every == 0:
            try:
                nfe_count = evaluate_training_nfe(net, ema, cfg, loaders, device)
                print(f"           | adaptive NFE {nfe_count}")
                if cfg.use_wandb:
                    wandb.log({"eval/adaptive_nfe": nfe_count, "step": step})
            except Exception as e:
                print(f"           | NFE evaluation failed: {e}")

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
