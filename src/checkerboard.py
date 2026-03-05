"""2D checkerboard dataset and MLP model for flow matching experiments (Figures 4, 9)."""
from __future__ import annotations
import torch
import torch.nn as nn


def sample_checkerboard(n: int, scale: float = 4.0) -> torch.Tensor:
    """Sample from a 2D 4x4 checkerboard distribution.

    Returns points in [-scale, scale]^2 on the 'black' squares.
    """
    # Pick a random black square (checkerboard pattern)
    # 4x4 grid => 8 black squares
    coords = []
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                coords.append((i, j))
    idx = torch.randint(0, len(coords), (n,))
    selected = torch.tensor(coords)[idx].float()  # (n, 2), integer grid coords

    # Uniform noise within each cell, then shift to [-scale, scale]
    cell_size = 2 * scale / 4
    noise = torch.rand(n, 2)
    points = (selected + noise) * cell_size - scale
    return points


class FlowMatchingMLP(nn.Module):
    """5-layer MLP with 512 hidden units, as described in the paper for 2D experiments."""

    def __init__(self, hidden: int = 512, num_layers: int = 5):
        super().__init__()
        layers = []
        in_dim = 3  # 2D point + 1D time
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden, 2))  # output 2D vector field
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 2) spatial coordinates
            t: (B,) time values
        Returns:
            (B, 2) predicted vector field
        """
        inp = torch.cat([x, t[:, None]], dim=-1)
        return self.net(inp)
