"""Sensor-specific feature encoders."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        depth: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        hidden = hidden_dim or max(output_dim, input_dim * 2)
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(depth - 1, 1)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden),
                    nn.GELU(),
                    nn.LayerNorm(hidden),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IMUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        depth: int = 2,
        dropout: float = 0.05,
        temporal_conv: bool = True,
        conv_kernel: int = 5,
    ) -> None:
        super().__init__()
        self.temporal_conv = temporal_conv
        hidden = hidden_dim or max(output_dim, input_dim * 2)
        if temporal_conv:
            pad = conv_kernel // 2
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, hidden, kernel_size=conv_kernel, padding=pad),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, kernel_size=conv_kernel, padding=pad),
                nn.GELU(),
            )
            mlp_input = hidden
        else:
            self.conv = None
            mlp_input = input_dim
        self.proj = MLPEncoder(mlp_input, output_dim, hidden_dim=hidden, depth=depth, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.proj(x)


class SensorEncoderBank(nn.Module):
    def __init__(
        self,
        wheel_dim: int,
        gps_dim: int,
        imu_dim: int,
        sensor_dim: int,
        dropout: float,
        imu_temporal_conv: bool = True,
    ) -> None:
        super().__init__()
        self.wheel = MLPEncoder(wheel_dim, sensor_dim, depth=2, dropout=dropout)
        self.gps = MLPEncoder(gps_dim, sensor_dim, hidden_dim=sensor_dim * 2, depth=3, dropout=dropout)
        self.imu = IMUEncoder(
            imu_dim,
            sensor_dim,
            hidden_dim=sensor_dim * 2,
            depth=2,
            dropout=dropout,
            temporal_conv=imu_temporal_conv,
        )

    def forward(self, wheel: torch.Tensor, gps: torch.Tensor, imu: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "wheel": self.wheel(wheel),
            "gps": self.gps(gps),
            "imu": self.imu(imu),
        }
