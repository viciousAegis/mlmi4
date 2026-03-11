from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer


class TokenChunkDataset(Dataset):
    """Fixed-length contiguous token chunks for unconditional text training."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        if tokens.ndim != 1:
            raise ValueError("tokens must be 1D")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self.tokens = tokens
        self.seq_len = seq_len
        self.num_chunks = tokens.numel() // seq_len
        if self.num_chunks == 0:
            raise ValueError("Not enough tokens to form one sequence")

    def __len__(self) -> int:
        return self.num_chunks

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = idx * self.seq_len
        end = start + self.seq_len
        return self.tokens[start:end]


@dataclass
class TextDataLoaders:
    train: DataLoader
    val: DataLoader
    tokenizer_name: str
    vocab_size: int
    seq_len: int


def _tokenize_lines(tokenizer: AutoTokenizer, lines: Iterable[str]) -> list[int]:
    all_ids: list[int] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        all_ids.extend(tokenizer.encode(s, add_special_tokens=False))
    return all_ids


def _load_text_tokens(
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_config: str | None,
    text_file: str | None,
    split: str,
    max_documents: int,
) -> torch.Tensor:
    if text_file is not None:
        with open(text_file, "r", encoding="utf-8") as f:
            ids = _tokenize_lines(tokenizer, f)
        return torch.tensor(ids, dtype=torch.long)

    from datasets import load_dataset

    ds = load_dataset(dataset_name, dataset_config, split=split)
    ids: list[int] = []
    for i, row in enumerate(ds):
        if i >= max_documents:
            break
        text = row.get("text", "")
        if not text:
            continue
        ids.extend(tokenizer.encode(text, add_special_tokens=False))

    if not ids:
        raise ValueError("No tokens loaded from dataset")
    return torch.tensor(ids, dtype=torch.long)


def get_text_dataloaders(
    tokenizer_name: str = "gpt2",
    dataset_name: str = "openwebtext",
    dataset_config: str | None = None,
    text_file: str | None = None,
    split: str = "train",
    max_documents: int = 100_000,
    seq_len: int = 256,
    batch_size: int = 64,
    val_ratio: float = 0.01,
    num_workers: int = 4,
    seed: int = 42,
) -> TextDataLoaders:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokens = _load_text_tokens(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        text_file=text_file,
        split=split,
        max_documents=max_documents,
    )

    full = TokenChunkDataset(tokens=tokens, seq_len=seq_len)
    val_size = max(1, int(len(full) * val_ratio))
    if val_size >= len(full):
        val_size = len(full) // 10
    train_size = len(full) - val_size
    if train_size <= 0:
        raise ValueError("Not enough data for train/val split")

    gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [train_size, val_size], generator=gen)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    return TextDataLoaders(
        train=train_loader,
        val=val_loader,
        tokenizer_name=tokenizer_name,
        vocab_size=int(tokenizer.vocab_size),
        seq_len=seq_len,
    )
