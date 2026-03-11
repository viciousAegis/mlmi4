from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys

import torch
from transformers import AutoTokenizer

from eval_text import eval_denoising, evaluate_generated_ppl, load_model, sample_sequences
from src.text_datasets import get_text_dataloaders


def parse_list_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_list_float(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def latest_ckpt(out_dir: str) -> str:
    paths = glob.glob(os.path.join(out_dir, "ckpt_step*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints found in {out_dir}")
    paths.sort(key=lambda p: int(os.path.basename(p).split("step")[1].split(".")[0]))
    return paths[-1]


def run_training(config: str | None, train_args: str | None) -> None:
    cmd = [sys.executable, "training_text.py"]
    if config is not None:
        cmd += ["--config", config]
    if train_args:
        cmd += train_args.split()
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Automate text DFM training + eval sweeps")
    p.add_argument("--train", action="store_true", help="Run training before evaluation")
    p.add_argument("--train-config", type=str, default=None)
    p.add_argument("--train-args", type=str, default=None, help="Extra args string for training_text.py")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="./runs/text_dfm")

    p.add_argument("--dataset-name", type=str, default="openwebtext")
    p.add_argument("--dataset-config", type=str, default=None)
    p.add_argument("--text-file", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-documents", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--eval-batches", type=int, default=200)

    p.add_argument("--nfe-list", type=str, default="64,128,256,512")
    p.add_argument("--temperature-list", type=str, default="0.8,0.9,1.0")
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--schedule", type=str, default="cubic", choices=["linear", "square", "cubic"])
    p.add_argument("--scorer-model", type=str, default="gpt2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--csv-out", type=str, default=None)

    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="mlmi4-text-dfm")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, nargs="+", default=None)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.train:
        run_training(args.train_config, args.train_args)

    ckpt_path = args.ckpt or latest_ckpt(args.out_dir)
    model, ckpt = load_model(ckpt_path, device)

    loaders = get_text_dataloaders(
        tokenizer_name=ckpt["tokenizer_name"],
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_file=args.text_file,
        split=args.split,
        max_documents=args.max_documents,
        seq_len=ckpt["seq_len"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    denoise_ce, denoise_acc = eval_denoising(
        model,
        loaders.val,
        mask_token_id=ckpt["mask_token_id"],
        schedule=args.schedule,
        eval_batches=args.eval_batches,
        device=device,
    )

    print(f"checkpoint={ckpt_path}")
    print(f"denoise_ce={denoise_ce:.6f}")
    print(f"denoise_masked_acc={denoise_acc:.6f}")

    run_rows: list[dict] = []

    if args.use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            config=vars(args),
        )
        wandb.log({"eval/denoise_ce": denoise_ce, "eval/denoise_masked_acc": denoise_acc})

    tok = AutoTokenizer.from_pretrained(ckpt["tokenizer_name"])
    for nfe in parse_list_int(args.nfe_list):
        for temp in parse_list_float(args.temperature_list):
            row = {
                "checkpoint": ckpt_path,
                "nfe": nfe,
                "temperature": temp,
                "denoise_ce": denoise_ce,
                "denoise_masked_acc": denoise_acc,
            }
            if args.scorer_model is not None:
                token_ids = sample_sequences(
                    model,
                    n=args.n_samples,
                    nfe=nfe,
                    temperature=temp,
                    schedule=args.schedule,
                    device=device,
                )
                texts = [tok.decode(t.tolist(), clean_up_tokenization_spaces=True) for t in token_ids]
                ppl = evaluate_generated_ppl(texts, args.scorer_model, device)
                row[f"generated_ppl_{args.scorer_model}"] = ppl
                print(f"nfe={nfe} temp={temp:.2f} ppl={ppl:.6f}")
            else:
                print(f"nfe={nfe} temp={temp:.2f} (no scorer model)")

            if args.use_wandb:
                wandb.log({f"sweep/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
            run_rows.append(row)

    csv_out = args.csv_out or os.path.join(args.out_dir, "text_eval_sweep.csv")
    os.makedirs(os.path.dirname(csv_out), exist_ok=True)
    fields = sorted({k for r in run_rows for k in r.keys()})
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(run_rows)

    print(f"saved_csv={csv_out}")

    if args.use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
