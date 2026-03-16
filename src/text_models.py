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


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    x_rot = torch.stack((-x_odd, x_even), dim=-1)
    return x_rot.flatten(start_dim=-2)


class RotaryAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0, rope_theta: float = 10_000.0):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by n_heads={n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim={self.head_dim} must be even for RoPE")
        self.dropout = dropout

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seqlen = q.shape[2]
        pos = torch.arange(seqlen, device=q.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1)[None, None, :, :]
        sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1)[None, None, :, :]
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        qkv = self.qkv(x).view(bsz, seqlen, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = self._apply_rope(q, k)
        h = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        h = h.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)
        return self.proj(h)


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0, rope_theta: float = 10_000.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = RotaryAttention(dim=dim, n_heads=n_heads, dropout=dropout, rope_theta=rope_theta)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa[:, None, :] * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp[:, None, :] * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiscreteFlowTransformer(nn.Module):
    """DiT-style transformer denoiser over masked token sequences with RoPE attention."""

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
        rope_theta: float = 10_000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.seq_len = seq_len

        self.token_emb = nn.Embedding(vocab_size + 1, dim)
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    n_heads=n_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                    rope_theta=rope_theta,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.final_modulation[-1].weight)
        nn.init.zeros_(self.final_modulation[-1].bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 2:
            raise ValueError("x_t must have shape (B, L)")
        if x_t.shape[1] > self.seq_len:
            raise ValueError(f"x_t length {x_t.shape[1]} exceeds configured seq_len {self.seq_len}")

        h = self.token_emb(x_t)
        c = self.time_emb(t.float())
        for block in self.blocks:
            h = block(h, c)
        shift, scale = self.final_modulation(c).chunk(2, dim=-1)
        h = modulate(self.final_norm(h), shift, scale)
        return F.linear(h, self.token_emb.weight[: self.vocab_size])
