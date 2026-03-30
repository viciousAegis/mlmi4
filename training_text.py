from __future__ import annotations

import argparse
import glob
import os
import re
import signal
import time

import torch
import torch.nn.functional as F
import wandb
import yaml

from src.text_datasets import get_text_dataloaders
from src.text_models import DiscreteFlowTransformer
from src.text_paths import discrete_corrupt_and_target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train discrete flow-matching text model")
    p.add_argument("--config", type=str, default=None, help="YAML config path")
    p.add_argument("--tokenizer", type=str, default="gpt2")
    p.add_argument("--dataset-name", type=str, default="openwebtext")
    p.add_argument("--dataset-config", type=str, default=None)
    p.add_argument("--text-file", type=str, default=None, help="Local plaintext fallback")
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-documents", type=int, default=100_000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--effective-batch", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--val-every", type=int, default=2_000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--ckpt-every", type=int, default=5_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=2_500)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--schedule", type=str, default="square", choices=["linear", "square", "cubic"])
    p.add_argument("--dim", type=int, default=768)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--ff-mult", type=int, default=4)
    p.add_argument("--rope-theta", type=float, default=10_000.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--use-amp", action="store_true")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="mlmi4-text-dfm")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, nargs="+", default=None)
    p.add_argument("--out-dir", type=str, default="./runs/text_dfm")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    p.add_argument("--auto-resume", action="store_true", help="Auto-resume from latest checkpoint in out-dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()

def apply_yaml_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


@torch.no_grad()
def evaluate(
    net: DiscreteFlowTransformer,
    loader,
    mask_token_id: int,
    schedule: str,
    device: str,
    max_batches: int = 50,
) -> tuple[float, float]:
    net.eval()
    losses: list[float] = []
    accs: list[float] = []

    it = iter(loader)
    for _ in range(max_batches):
        try:
            x1 = next(it)
        except StopIteration:
            break

        x1 = x1.to(device)
        t, x_t, target_mask = discrete_corrupt_and_target(x1, mask_token_id=mask_token_id, schedule=schedule)
        logits = net(x_t, t)

        masked_count = target_mask.sum().item()
        if masked_count == 0:
            continue

        loss = F.cross_entropy(logits[target_mask], x1[target_mask])
        pred = logits.argmax(dim=-1)
        acc = (pred[target_mask] == x1[target_mask]).float().mean().item()

        losses.append(float(loss.item()))
        accs.append(float(acc))

    net.train()
    if not losses:
        return 0.0, 0.0
    return float(sum(losses) / len(losses)), float(sum(accs) / len(accs))


def _lr_at_step(step: int, peak_lr: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return peak_lr
    return peak_lr * min(1.0, step / float(warmup_steps))


def _set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def _latest_checkpoint_path(out_dir: str) -> str | None:
    candidates = glob.glob(os.path.join(out_dir, "ckpt_step*.pt"))
    if not candidates:
        return None
    pat = re.compile(r"ckpt_step(\d+)\.pt$")
    ranked: list[tuple[int, str]] = []
    for path in candidates:
        m = pat.search(path)
        if m is None:
            continue
        ranked.append((int(m.group(1)), path))
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0])
    return ranked[-1][1]


def main() -> None:
    args = apply_yaml_overrides(parse_args())
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    loaders = get_text_dataloaders(
        tokenizer_name=args.tokenizer,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_file=args.text_file,
        split=args.split,
        max_documents=args.max_documents,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    mask_token_id = loaders.vocab_size
    net = DiscreteFlowTransformer(
        vocab_size=loaders.vocab_size,
        mask_token_id=mask_token_id,
        seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_mult=args.ff_mult,
        rope_theta=args.rope_theta,
        dropout=args.dropout,
    ).to(device)

    opt = torch.optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    use_amp = args.use_amp and device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    autocast = torch.amp.autocast

    accum_steps = max(1, args.effective_batch // args.batch_size)
    train_it = iter(loaders.train)
    t0 = time.time()
    start_step = 0

    print(f"device={device}")
    print(f"vocab_size={loaders.vocab_size}, seq_len={args.seq_len}, schedule={args.schedule}")
    if args.effective_batch % args.batch_size != 0:
        print(
            f"[warn] effective_batch={args.effective_batch} not divisible by batch_size={args.batch_size}; "
            f"using accum_steps={accum_steps}"
        )

    resume_path = args.resume
    if resume_path is None and args.auto_resume:
        resume_path = _latest_checkpoint_path(args.out_dir)
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location="cpu")
        net.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from: {resume_path} (step={start_step})")

    stop_requested = False

    def _request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"\nreceived signal {signum}; will checkpoint and stop after current step")

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sigint = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    def _save_checkpoint(step: int) -> str:
        ckpt_path = os.path.join(args.out_dir, f"ckpt_step{step}.pt")
        torch.save(
            {
                "step": step,
                "model": net.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict() if use_amp else None,
                "args": vars(args),
                "vocab_size": loaders.vocab_size,
                "mask_token_id": mask_token_id,
                "tokenizer_name": loaders.tokenizer_name,
                "seq_len": args.seq_len,
            },
            ckpt_path,
        )
        return ckpt_path

    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config=vars(args),
        )
        if start_step > 0:
            wandb.log({"resume/step": start_step}, step=start_step)

    if start_step >= args.total_steps:
        print(f"resume step {start_step} already reached total_steps={args.total_steps}; exiting")
        if args.use_wandb:
            wandb.finish()
        signal.signal(signal.SIGTERM, previous_sigterm)
        signal.signal(signal.SIGINT, previous_sigint)
        return

    for step in range(start_step + 1, args.total_steps + 1):
        lr = _lr_at_step(step=step, peak_lr=args.lr, warmup_steps=args.warmup_steps)
        _set_lr(opt, lr)
        opt.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_masked = 0

        for _ in range(accum_steps):
            try:
                x1 = next(train_it)
            except StopIteration:
                train_it = iter(loaders.train)
                x1 = next(train_it)

            x1 = x1.to(device, non_blocking=(device == "cuda"))
            t, x_t, target_mask = discrete_corrupt_and_target(
                x1,
                mask_token_id=mask_token_id,
                schedule=args.schedule,
            )

            masked_count = target_mask.sum().item()
            if masked_count == 0:
                continue

            if use_amp:
                with autocast("cuda", dtype=torch.float16):
                    logits = net(x_t, t)
                    loss = F.cross_entropy(logits[target_mask], x1[target_mask]) / accum_steps
                scaler.scale(loss).backward()
            else:
                logits = net(x_t, t)
                loss = F.cross_entropy(logits[target_mask], x1[target_mask]) / accum_steps
                loss.backward()

            total_loss += float(loss.item())
            total_masked += int(masked_count)

        if use_amp:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        if step % args.log_every == 0 or step == 1:
            steps_per_s = step / max(1e-6, time.time() - t0)
            print(
                f"step {step:07d} | lr {lr:.3e} | train_loss {total_loss:.6f} | "
                f"masked_tok {total_masked} | {steps_per_s:.2f} steps/s"
            )
            if args.use_wandb:
                wandb.log(
                    {
                        "train/lr": lr,
                        "train/loss": total_loss,
                        "train/masked_tokens": total_masked,
                        "train/steps_per_s": steps_per_s,
                        "step": step,
                    }
                )

        if step % args.val_every == 0:
            val_loss, val_acc = evaluate(
                net,
                loaders.val,
                mask_token_id=mask_token_id,
                schedule=args.schedule,
                device=device,
            )
            print(f"             | val_loss {val_loss:.6f} | val_masked_acc {val_acc:.4f}")
            if args.use_wandb:
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/masked_acc": val_acc,
                        "step": step,
                    }
                )

        if step % args.ckpt_every == 0 or step == args.total_steps:
            ckpt_path = _save_checkpoint(step)
            print(f"saved: {ckpt_path}")

        if stop_requested:
            ckpt_path = _save_checkpoint(step)
            print(f"saved interrupt checkpoint: {ckpt_path}")
            break

    if args.use_wandb:
        wandb.finish()
    signal.signal(signal.SIGTERM, previous_sigterm)
    signal.signal(signal.SIGINT, previous_sigint)


if __name__ == "__main__":
    main()
