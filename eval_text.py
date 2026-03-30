from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.text_datasets import get_text_dataloaders
from src.text_models import DiscreteFlowTransformer
from src.text_paths import discrete_corrupt_and_target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate discrete FM text model")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--dataset-name", type=str, default="openwebtext")
    p.add_argument("--dataset-config", type=str, default=None)
    p.add_argument("--text-file", type=str, default=None)
    p.add_argument("--split", type=str, default="train")
    p.add_argument("--max-documents", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--schedule", type=str, default="square", choices=["linear", "square", "cubic"])
    p.add_argument("--eval-batches", type=int, default=200)
    p.add_argument("--scorer-model", type=str, default="gpt2", help="LM used for generated-text PPL")
    p.add_argument("--n-samples", type=int, default=128)
    p.add_argument("--nfe", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--temperature-scheduler",
        type=str,
        default="paper",
        choices=["constant", "paper"],
        help="paper: tau_t = tau * (1 - t)^2",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_model(ckpt_path: str, device: str) -> tuple[DiscreteFlowTransformer, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]
    model = DiscreteFlowTransformer(
        vocab_size=ckpt["vocab_size"],
        mask_token_id=ckpt["mask_token_id"],
        seq_len=ckpt["seq_len"],
        dim=args["dim"],
        n_layers=args["n_layers"],
        n_heads=args["n_heads"],
        ff_mult=args["ff_mult"],
        rope_theta=args.get("rope_theta", 10_000.0),
        dropout=args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


@torch.no_grad()
def eval_denoising(
    model: DiscreteFlowTransformer,
    loader,
    mask_token_id: int,
    schedule: str,
    eval_batches: int,
    device: str,
) -> tuple[float, float]:
    losses: list[float] = []
    accs: list[float] = []

    it = iter(loader)
    for _ in range(eval_batches):
        try:
            x1 = next(it)
        except StopIteration:
            break

        x1 = x1.to(device)
        t, x_t, target_mask = discrete_corrupt_and_target(x1, mask_token_id=mask_token_id, schedule=schedule)
        logits = model(x_t, t)

        if target_mask.sum().item() == 0:
            continue

        loss = F.cross_entropy(logits[target_mask], x1[target_mask]).item()
        pred = logits.argmax(dim=-1)
        acc = (pred[target_mask] == x1[target_mask]).float().mean().item()

        losses.append(float(loss))
        accs.append(float(acc))

    if not losses:
        return 0.0, 0.0
    return float(sum(losses) / len(losses)), float(sum(accs) / len(accs))


@torch.no_grad()
def sample_sequences(
    model: DiscreteFlowTransformer,
    n: int,
    nfe: int,
    temperature: float,
    temperature_scheduler: str,
    schedule: str,
    device: str,
) -> torch.Tensor:
    from src.text_paths import reveal_probability

    seq_len = model.seq_len
    mask_token_id = model.mask_token_id
    x = torch.full((n, seq_len), mask_token_id, dtype=torch.long, device=device)

    ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
    for i in range(nfe):
        t_cur = ts[i].expand(n)
        t_next = ts[i + 1].expand(n)
        logits = model(x, t_cur)
        if temperature_scheduler == "paper":
            tau_t = temperature * (1.0 - t_cur) ** 2
            logits = logits / tau_t[:, None, None].clamp_min(1e-6)
        else:
            logits = logits / max(1e-6, temperature)

        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1).view(n, seq_len)

        r = reveal_probability(t_cur, t_next, schedule=schedule)
        reveal = (torch.rand_like(x, dtype=torch.float32) < r[:, None]) & x.eq(mask_token_id)
        x = torch.where(reveal, sampled, x)

    remaining = x.eq(mask_token_id)
    if remaining.any():
        logits = model(x, torch.ones(n, device=device))
        x = torch.where(remaining, logits.argmax(dim=-1), x)

    return x


@torch.no_grad()
def evaluate_generated_ppl(
    generated_texts: list[str],
    scorer_model_name: str,
    device: str,
) -> float:
    scorer_tok = AutoTokenizer.from_pretrained(scorer_model_name)
    scorer = AutoModelForCausalLM.from_pretrained(scorer_model_name).to(device)
    scorer.eval()

    losses: list[float] = []
    for text in generated_texts:
        ids = scorer_tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        if ids.numel() < 2:
            continue
        out = scorer(input_ids=ids, labels=ids)
        losses.append(float(out.loss.item()))

    if not losses:
        return float("nan")
    mean_nll = sum(losses) / len(losses)
    return float(math.exp(mean_nll))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_model(args.ckpt, device)

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

    loss, acc = eval_denoising(
        model,
        loaders.val,
        mask_token_id=ckpt["mask_token_id"],
        schedule=args.schedule,
        eval_batches=args.eval_batches,
        device=device,
    )
    print(f"denoise_ce={loss:.6f}")
    print(f"denoise_masked_acc={acc:.6f}")

    if args.scorer_model is not None:
        token_ids = sample_sequences(
            model,
            n=args.n_samples,
            nfe=args.nfe,
            temperature=args.temperature,
            temperature_scheduler=args.temperature_scheduler,
            schedule=args.schedule,
            device=device,
        )
        tok = AutoTokenizer.from_pretrained(ckpt["tokenizer_name"])
        texts = [tok.decode(row.tolist(), clean_up_tokenization_spaces=True) for row in token_ids]
        ppl = evaluate_generated_ppl(texts, args.scorer_model, device)
        print(f"generated_ppl_{args.scorer_model}={ppl:.6f}")


if __name__ == "__main__":
    main()
