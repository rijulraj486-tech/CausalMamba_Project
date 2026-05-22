"""Pure Mamba neural filtering architecture for NCLT navigation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.filter_blocks import (
    FilterFusion,
    FilterSensorEncoders,
    MambaLatentTransition,
    MLP,
    NeuralCorrectionBlock,
    StateDecoder,
)


@dataclass
class PureMambaFilterConfig:
    wheel_dim: int = 7
    gps_dim: int = 10
    imu_dim: int = 11
    sensor_dim: int = 128
    latent_dim: int = 256
    decoder_hidden_dim: int = 384
    transition_layers: int = 6
    d_state: int = 32
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.05
    variant: str = "mamba"
    bidirectional: bool = False
    fusion: str = "gated"
    fusion_heads: int = 4
    imu_temporal_conv: bool = True
    initial_state_dim: int = 6
    output_state_dim: int = 8
    probabilistic: bool = False


class PureMambaFilter(nn.Module):
    """End-to-end learned predict-correct style Mamba filter.

    This module uses no Kalman equations or classical filtering updates. The
    transition and correction are learned latent-space neural modules.
    """

    def __init__(self, cfg: PureMambaFilterConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoders = FilterSensorEncoders(
            wheel_dim=cfg.wheel_dim,
            gps_dim=cfg.gps_dim,
            imu_dim=cfg.imu_dim,
            embed_dim=cfg.sensor_dim,
            dropout=cfg.dropout,
            imu_temporal_conv=cfg.imu_temporal_conv,
        )
        self.fusion = FilterFusion(
            embed_dim=cfg.sensor_dim,
            latent_dim=cfg.latent_dim,
            mode=cfg.fusion,
            num_heads=cfg.fusion_heads,
            dropout=cfg.dropout,
        )
        self.input_proj = nn.Sequential(
            nn.LayerNorm(cfg.latent_dim),
            nn.Linear(cfg.latent_dim, cfg.latent_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.initial_encoder = MLP(
            cfg.initial_state_dim,
            cfg.latent_dim,
            hidden_dim=cfg.latent_dim,
            depth=3,
            dropout=cfg.dropout,
        )
        self.transition = MambaLatentTransition(
            latent_dim=cfg.latent_dim,
            layers=cfg.transition_layers,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            dropout=cfg.dropout,
            variant=cfg.variant,
            bidirectional=cfg.bidirectional,
        )
        self.correction = NeuralCorrectionBlock(cfg.latent_dim, cfg.sensor_dim, dropout=cfg.dropout)
        self.decoder = StateDecoder(
            latent_dim=cfg.latent_dim,
            hidden_dim=cfg.decoder_hidden_dim,
            state_dim=cfg.output_state_dim,
            dropout=cfg.dropout,
            probabilistic=cfg.probabilistic,
        )

    def forward(
        self,
        wheel: torch.Tensor,
        gps: torch.Tensor,
        imu: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoders(wheel, gps, imu)
        fused, gates = self.fusion(encoded)
        initial_context = self.initial_encoder(torch.nan_to_num(initial_state.float(), nan=0.0))
        latent_input = self.input_proj(fused) + initial_context[:, None, :]
        predicted_latent = self.transition(latent_input)
        corrected_latent = self.correction(
            predicted_latent=predicted_latent,
            gps_embed=encoded["gps"],
            imu_embed=encoded["imu"],
            gps_mask=encoded["gps_mask"],
        )
        decoded = self.decoder(corrected_latent)
        decoded.update(
            {
                "state_estimates": decoded["state"],
                "sensor_gates": gates,
                "predicted_latent": predicted_latent,
                "corrected_latent": corrected_latent,
            }
        )
        return decoded


def build_pure_mamba_filter(cfg: PureMambaFilterConfig) -> PureMambaFilter:
    return PureMambaFilter(cfg)
