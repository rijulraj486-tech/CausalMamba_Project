"""Neural building blocks for the pure Mamba filtering path."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.mamba_blocks import MambaBlockConfig, MambaStack


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, depth: int = 2, dropout: float = 0.05) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(depth - 1, 1)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WheelEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = MLP(input_dim, embed_dim, hidden_dim=embed_dim * 2, depth=3, dropout=dropout)

    def forward(self, wheel: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(wheel.float(), nan=0.0, posinf=0.0, neginf=0.0))


class GPSEncoder(nn.Module):
    """GPS encoder with an explicit availability mask embedding.

    The last input channel is expected to be a finite-GPS availability mask from
    the stable NCLT feature pipeline.
    """

    def __init__(self, input_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.feature_net = MLP(input_dim, embed_dim, hidden_dim=embed_dim * 2, depth=3, dropout=dropout)
        self.mask_net = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, gps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gps = torch.nan_to_num(gps.float(), nan=0.0, posinf=0.0, neginf=0.0)
        mask = gps[..., -1:].clamp(0.0, 1.0)
        features = self.feature_net(gps)
        mask_emb = self.mask_net(mask)
        return self.out(torch.cat([features, mask_emb], dim=-1)), mask


class IMUEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, dropout: float, temporal_conv: bool = True) -> None:
        super().__init__()
        hidden_dim = embed_dim * 2
        if temporal_conv:
            self.temporal = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
            )
            mlp_input = hidden_dim
        else:
            self.temporal = None
            mlp_input = input_dim
        self.net = MLP(mlp_input, embed_dim, hidden_dim=hidden_dim, depth=2, dropout=dropout)

    def forward(self, imu: torch.Tensor) -> torch.Tensor:
        imu = torch.nan_to_num(imu.float(), nan=0.0, posinf=0.0, neginf=0.0)
        if self.temporal is not None:
            imu = self.temporal(imu.transpose(1, 2)).transpose(1, 2)
        return self.net(imu)


class FilterSensorEncoders(nn.Module):
    def __init__(self, wheel_dim: int, gps_dim: int, imu_dim: int, embed_dim: int, dropout: float, imu_temporal_conv: bool) -> None:
        super().__init__()
        self.wheel = WheelEncoder(wheel_dim, embed_dim, dropout)
        self.gps = GPSEncoder(gps_dim, embed_dim, dropout)
        self.imu = IMUEncoder(imu_dim, embed_dim, dropout, temporal_conv=imu_temporal_conv)

    def forward(self, wheel: torch.Tensor, gps: torch.Tensor, imu: torch.Tensor) -> dict[str, torch.Tensor]:
        gps_embed, gps_mask = self.gps(gps)
        return {
            "wheel": self.wheel(wheel),
            "gps": gps_embed,
            "imu": self.imu(imu),
            "gps_mask": gps_mask,
        }


class FilterFusion(nn.Module):
    def __init__(self, embed_dim: int, latent_dim: int, mode: str = "gated", num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.mode = mode
        self.gate_net = nn.Sequential(
            nn.LayerNorm(embed_dim * 3 + 1),
            nn.Linear(embed_dim * 3 + 1, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 3),
        )
        self.gated_out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.concat_out = nn.Sequential(
            nn.LayerNorm(embed_dim * 3 + 1),
            nn.Linear(embed_dim * 3 + 1, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.attn_out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, encoded: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        wheel = encoded["wheel"]
        gps = encoded["gps"]
        imu = encoded["imu"]
        gps_mask = encoded["gps_mask"]
        cat = torch.cat([wheel, gps, imu, gps_mask], dim=-1)
        if self.mode == "concat":
            gates = wheel.new_full((*wheel.shape[:2], 3), 1.0 / 3.0)
            return self.concat_out(cat), gates
        if self.mode == "cross_attention":
            bsz, seq_len, dim = wheel.shape
            sensors = torch.stack([wheel, gps, imu], dim=2).reshape(bsz * seq_len, 3, dim)
            query = self.query.expand(bsz * seq_len, 1, dim)
            fused, weights = self.attn(query, sensors, sensors, need_weights=True)
            return self.attn_out(fused.reshape(bsz, seq_len, dim)), weights.reshape(bsz, seq_len, 3)
        if self.mode != "gated":
            raise ValueError(f"Unknown fusion mode: {self.mode}")
        logits = self.gate_net(cat)
        logits = logits.clone()
        logits[..., 1:2] = logits[..., 1:2] + torch.log(gps_mask.clamp_min(1.0e-4))
        gates = torch.softmax(logits, dim=-1)
        stacked = torch.stack([wheel, gps, imu], dim=-2)
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=-2)
        return self.gated_out(fused), gates


class MambaLatentTransition(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        layers: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        variant: str,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        self.stack = MambaStack(
            MambaBlockConfig(
                dim=latent_dim,
                layers=layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                variant=variant,
                bidirectional=bidirectional,
            )
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.stack(latent)


class NeuralCorrectionBlock(nn.Module):
    """Purely neural gated residual correction in latent space."""

    def __init__(self, latent_dim: int, sensor_dim: int, dropout: float) -> None:
        super().__init__()
        input_dim = latent_dim + sensor_dim * 2 + 1
        self.delta = MLP(input_dim, latent_dim, hidden_dim=latent_dim * 2, depth=3, dropout=dropout)
        self.gate = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, predicted_latent: torch.Tensor, gps_embed: torch.Tensor, imu_embed: torch.Tensor, gps_mask: torch.Tensor) -> torch.Tensor:
        context = torch.cat([predicted_latent, gps_embed, imu_embed, gps_mask], dim=-1)
        corrected = predicted_latent + self.gate(context) * self.delta(context)
        return self.norm(corrected)


class StateDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, state_dim: int = 8, dropout: float = 0.05, probabilistic: bool = False) -> None:
        super().__init__()
        self.probabilistic = probabilistic
        self.mean = MLP(latent_dim, state_dim, hidden_dim=hidden_dim, depth=3, dropout=dropout)
        self.log_var = MLP(latent_dim, 2, hidden_dim=hidden_dim, depth=2, dropout=dropout) if probabilistic else None

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        state = self.mean(latent)
        output = {
            "state": state,
            "position": state[..., :2],
        }
        if self.log_var is not None:
            output["log_var"] = self.log_var(latent).clamp(-8.0, 6.0)
        return output
