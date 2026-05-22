"""Reliability-aware sensor fusion layers."""

from __future__ import annotations

import torch
import torch.nn as nn


class ReliabilityGatedFusion(nn.Module):
    def __init__(self, sensor_dim: int, output_dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.sensor_names = ("wheel", "gps", "imu")
        self.reliability = nn.Sequential(
            nn.LayerNorm(sensor_dim * 3),
            nn.Linear(sensor_dim * 3, sensor_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sensor_dim, 3),
        )
        trust_hidden = max(sensor_dim // 2, 16)
        self.validity_trust = nn.Sequential(
            nn.Linear(3, trust_hidden),
            nn.GELU(),
            nn.Linear(trust_hidden, 3),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(sensor_dim),
            nn.Linear(sensor_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        sensor_validity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wheel, gps, imu = encoded["wheel"], encoded["gps"], encoded["imu"]
        logits = self.reliability(torch.cat([wheel, gps, imu], dim=-1))
        if sensor_validity is not None:
            logits = logits + self.validity_trust(sensor_validity)
        gates = torch.softmax(logits, dim=-1)
        stacked = torch.stack([wheel, gps, imu], dim=-2)
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=-2)
        return self.out(fused), gates


class CrossAttentionFusion(nn.Module):
    def __init__(self, sensor_dim: int, output_dim: int, num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=sensor_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, sensor_dim) * 0.02)
        self.out = nn.Sequential(
            nn.LayerNorm(sensor_dim),
            nn.Linear(sensor_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        sensor_validity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = encoded["wheel"].shape
        sensors = torch.stack([encoded["wheel"], encoded["gps"], encoded["imu"]], dim=2)
        sensors = sensors.reshape(bsz * seq_len, 3, dim)
        query = self.query.expand(bsz * seq_len, 1, dim)
        fused, weights = self.attn(query, sensors, sensors, need_weights=True)
        fused = fused.reshape(bsz, seq_len, dim)
        weights = weights.reshape(bsz, seq_len, 3)
        return self.out(fused), weights


class SensorFusion(nn.Module):
    def __init__(
        self,
        sensor_dim: int,
        output_dim: int,
        mode: str = "gated",
        num_heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.gated = ReliabilityGatedFusion(sensor_dim, output_dim, dropout=dropout)
        self.cross = CrossAttentionFusion(sensor_dim, output_dim, num_heads=num_heads, dropout=dropout)
        self.concat = nn.Sequential(
            nn.LayerNorm(sensor_dim * 3),
            nn.Linear(sensor_dim * 3, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if mode == "hybrid":
            self.hybrid = nn.Sequential(
                nn.LayerNorm(output_dim * 3),
                nn.Linear(output_dim * 3, output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.hybrid = None

    def forward(
        self,
        encoded: dict[str, torch.Tensor],
        sensor_validity: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "gated":
            return self.gated(encoded, sensor_validity=sensor_validity)
        if self.mode == "cross_attention":
            return self.cross(encoded, sensor_validity=sensor_validity)
        if self.mode == "concat":
            gates = encoded["wheel"].new_full((*encoded["wheel"].shape[:2], 3), 1.0 / 3.0)
            fused = self.concat(torch.cat([encoded["wheel"], encoded["gps"], encoded["imu"]], dim=-1))
            return fused, gates
        if self.mode == "hybrid":
            gated, gates = self.gated(encoded, sensor_validity=sensor_validity)
            cross, _ = self.cross(encoded, sensor_validity=sensor_validity)
            concat = self.concat(torch.cat([encoded["wheel"], encoded["gps"], encoded["imu"]], dim=-1))
            assert self.hybrid is not None
            return self.hybrid(torch.cat([gated, cross, concat], dim=-1)), gates
        raise ValueError(f"Unknown fusion mode: {self.mode}")
