from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from src.text_models import DiscreteFlowTransformer
from src.text_paths import reveal_probability


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample text from discrete flow-matching model")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--nfe", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--temperature-scheduler",
        type=str,
        default="paper",
        choices=["constant", "paper"],
        help="paper: tau_t = tau * (1 - t)^2",
    )
    p.add_argument("--schedule", type=str, default="square", choices=["linear", "square", "cubic"])
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_model(ckpt_path: str, device: str) -> tuple[DiscreteFlowTransformer, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt["args"]

    net = DiscreteFlowTransformer(
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
    net.load_state_dict(ckpt["model"], strict=True)
    net.eval()
    return net, ckpt


@torch.no_grad()
def sample_sequences(
    net: DiscreteFlowTransformer,
    n: int,
    nfe: int,
    temperature: float,
    temperature_scheduler: str,
    schedule: str,
    device: str,
) -> torch.Tensor:
    seq_len = net.seq_len
    mask_token_id = net.mask_token_id
    x = torch.full((n, seq_len), fill_value=mask_token_id, dtype=torch.long, device=device)

    ts = torch.linspace(0.0, 1.0, nfe + 1, device=device)
    for i in range(nfe):
        t_cur = ts[i].expand(n)
        t_next = ts[i + 1].expand(n)

        logits = net(x, t_cur)
        if temperature_scheduler == "paper":
            tau_t = temperature * (1.0 - t_cur) ** 2
            logits = logits / tau_t[:, None, None].clamp_min(1e-6)
        else:
            logits = logits / max(1e-6, temperature)
        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.reshape(-1, probs.size(-1)), num_samples=1).view(n, seq_len)

        r = reveal_probability(t_cur, t_next, schedule=schedule)
        reveal = torch.rand_like(x, dtype=torch.float32) < r[:, None]
        is_masked = x.eq(mask_token_id)
        update_pos = reveal & is_masked

        x = torch.where(update_pos, sampled, x)

    remaining = x.eq(mask_token_id)
    if remaining.any():
        t_final = torch.ones(n, device=device)
        logits = net(x, t_final)
        x = torch.where(remaining, logits.argmax(dim=-1), x)

    return x


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    net, ckpt = load_model(args.ckpt, device)

    tokenizer = AutoTokenizer.from_pretrained(ckpt["tokenizer_name"])
    tokens = sample_sequences(
        net=net,
        n=args.n,
        nfe=args.nfe,
        temperature=args.temperature,
        temperature_scheduler=args.temperature_scheduler,
        schedule=args.schedule,
        device=device,
    )

    if args.max_new_tokens is not None:
        tokens = tokens[:, : args.max_new_tokens]

    for i in range(tokens.shape[0]):
        text = tokenizer.decode(tokens[i].tolist(), clean_up_tokenization_spaces=True)
        print(f"[{i}] {text}\n")


if __name__ == "__main__":
    main()
