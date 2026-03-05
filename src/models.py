# model.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Time embedding
# ----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(0, half, device=t.device, dtype=t.dtype)
            / (half - 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, emb_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _groupnorm(ch: int) -> nn.GroupNorm:
    groups = 32 if ch >= 32 else max(1, ch // 4)
    return nn.GroupNorm(groups, ch)


# ----------------------------
# Blocks
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_ch: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _groupnorm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.tproj = nn.Linear(t_ch, out_ch)

        self.norm2 = _groupnorm(out_ch)
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.tproj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        assert ch % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = ch // num_heads

        self.norm = _groupnorm(ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=1)

        T = H * W
        q = q.view(B, self.num_heads, self.head_dim, T).transpose(
            2, 3
        )  # (B, heads, T, D)
        k = k.view(B, self.num_heads, self.head_dim, T)  # (B, heads, D, T)
        v = v.view(B, self.num_heads, self.head_dim, T).transpose(
            2, 3
        )  # (B, heads, T, D)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.softmax((q @ k) * scale, dim=-1)  # (B, heads, T, T)
        out = attn @ v  # (B, heads, T, D)

        out = out.transpose(2, 3).contiguous().view(B, C, H, W)
        out = self.proj(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ----------------------------
# U-Net (structured)
# ----------------------------
class UNetCIFAR(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        out_ch: int = 3,
        base_ch: int = 256,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions=(16,),
        num_heads: int = 4,
        dropout: float = 0.0,
        image_size: int = 32,
    ):
        super().__init__()
        self.attn_resolutions = set(attn_resolutions)
        time_emb_dim = base_ch
        t_ch = base_ch * 4

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            TimeMLP(time_emb_dim, t_ch),
        )

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down: per level modules
        self.down_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_ch
        resolution = image_size
        skip_chs: list[int] = []

        for i, mult in enumerate(channel_mults):
            outc = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(ch, outc, t_ch, dropout))
                ch = outc
                if resolution in self.attn_resolutions:
                    blocks.append(AttentionBlock(ch, num_heads=num_heads))
                skip_chs.append(ch)
            self.down_levels.append(blocks)

            if i != len(channel_mults) - 1:
                self.downsamples.append(Downsample(ch))
                resolution //= 2
            else:
                self.downsamples.append(nn.Identity())

        # Middle
        self.mid = nn.ModuleList(
            [
                ResBlock(ch, ch, t_ch, dropout),
                AttentionBlock(ch, num_heads=num_heads),
                ResBlock(ch, ch, t_ch, dropout),
            ]
        )

        # Up: per level modules
        self.up_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_mults))):
            outc = base_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                skip_ch = skip_chs.pop()
                blocks.append(ResBlock(ch + skip_ch, outc, t_ch, dropout))
                ch = outc
                if resolution in self.attn_resolutions:
                    blocks.append(AttentionBlock(ch, num_heads=num_heads))
            self.up_levels.append(blocks)

            if i != 0:
                self.upsamples.append(Upsample(ch))
                resolution *= 2
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = _groupnorm(ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time_embed(t)  # (B, t_ch)
        h = self.in_conv(x)

        skips = []
        for level, down in zip(self.down_levels, self.downsamples):
            for block in level:
                if isinstance(block, ResBlock):
                    h = block(h, temb)
                else:
                    h = block(h)
                # store after every block (matches skip list we built)
                if isinstance(block, ResBlock):
                    skips.append(h)
            h = down(h)

        # mid
        h = self.mid[0](h, temb)
        h = self.mid[1](h)
        h = self.mid[2](h, temb)

        # up
        for level, up in zip(self.up_levels, self.upsamples):
            for block in level:
                if isinstance(block, ResBlock):
                    h = torch.cat([h, skips.pop()], dim=1)
                    h = block(h, temb)
                else:
                    h = block(h)
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = UNetCIFAR().to(device)
    x = torch.randn(4, 3, 32, 32, device=device)
    t = torch.rand(4, device=device)
    y = net(x, t)
    print("out:", y.shape, y.dtype)
