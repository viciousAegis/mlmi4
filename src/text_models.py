from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device, dtype=t.dtype) / max(1, half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class DiscreteFlowTransformer(nn.Module):
    """Transformer denoiser over masked token sequences with time conditioning."""

    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        seq_len: int,
        dim: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(vocab_size + 1, dim)
        self.pos_emb = nn.Embedding(seq_len, dim)
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=ff_mult * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 2:
            raise ValueError("x_t must have shape (B, L)")
        if x_t.shape[1] > self.seq_len:
            raise ValueError(f"x_t length {x_t.shape[1]} exceeds configured seq_len {self.seq_len}")

        bsz, seqlen = x_t.shape
        pos = torch.arange(seqlen, device=x_t.device)

        h = self.token_emb(x_t) + self.pos_emb(pos)[None, :, :]
        t_cond = self.time_emb(t.float())[:, None, :]
        h = h + t_cond
        h = self.backbone(h)
        h = self.norm(h)
        return self.out(h)
