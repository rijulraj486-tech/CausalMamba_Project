"""Reusable forward-only Mamba temporal blocks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba, Mamba2  # type: ignore
except ImportError:  # pragma: no cover - dependency is expected in benchmark env.
    Mamba = None  # type: ignore
    Mamba2 = None  # type: ignore


@dataclass(frozen=True)
class MambaBlockConfig:
    dim: int = 192
    layers: int = 2
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.05
    variant: str = "mamba"
    bidirectional: bool = False


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_mamba(dim: int, cfg: MambaBlockConfig) -> nn.Module:
    if cfg.variant == "mamba2" and Mamba2 is not None:
        return Mamba2(d_model=dim, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)
    if Mamba is None:
        raise RuntimeError("mamba_ssm is required for pure Mamba navigation models.")
    return Mamba(d_model=dim, d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)


class MambaResidualBlock(nn.Module):
    """Pre-norm residual Mamba block with a small FFN."""

    def __init__(self, cfg: MambaBlockConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(cfg.dim)
        self.mixer = _make_mamba(cfg.dim, cfg)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ff = FeedForward(cfg.dim, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.mixer(self.norm(x)))
        x = x + self.ff(x)
        return x


class MambaStack(nn.Module):
    def __init__(self, cfg: MambaBlockConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([MambaResidualBlock(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class HierarchicalBiMamba(nn.Module):
    """Three-level forward-only temporal model kept for checkpoint compatibility."""

    def __init__(
        self,
        dim: int,
        local_layers: int = 2,
        mid_layers: int = 2,
        long_layers: int = 2,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.05,
        variant: str = "mamba",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        base = dict(
            dim=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            variant=variant,
            bidirectional=bidirectional,
        )
        self.local = MambaStack(MambaBlockConfig(layers=local_layers, **base))
        self.mid = MambaStack(MambaBlockConfig(layers=mid_layers, **base))
        self.long = MambaStack(MambaBlockConfig(layers=long_layers, **base))
        self.level_fusion = nn.Sequential(
            nn.LayerNorm(dim * 3),
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        mid = self.mid(local)
        long = self.long(mid)
        fused = self.level_fusion(torch.cat([local, mid, long], dim=-1))
        return self.out_norm(x + fused)
