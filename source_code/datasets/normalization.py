"""Per-channel normalization for sensor and trajectory features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

NormMode = Literal["zscore", "robust", "minmax", "none"]
MIN_SCALE = 1e-6
NORMALIZED_CLAMP = 50.0


def _safe_scale(scale: torch.Tensor) -> torch.Tensor:
    """Avoid epsilon-amplifying constant or almost-constant channels."""
    scale = torch.nan_to_num(scale.float(), nan=1.0, posinf=1.0, neginf=1.0)
    return torch.where(scale > MIN_SCALE, scale, torch.ones_like(scale))


@dataclass
class ChannelStats:
    mode: NormMode
    center: torch.Tensor
    scale: torch.Tensor
    min_value: torch.Tensor | None = None
    max_value: torch.Tensor | None = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        center = self.center.to(device=x.device, dtype=x.dtype)
        scale = _safe_scale(self.scale).to(device=x.device, dtype=x.dtype)
        normalized = (x - center) / scale
        normalized = torch.nan_to_num(normalized, nan=0.0, posinf=NORMALIZED_CLAMP, neginf=-NORMALIZED_CLAMP)
        return normalized.clamp(min=-NORMALIZED_CLAMP, max=NORMALIZED_CLAMP)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return x
        center = self.center.to(device=x.device, dtype=x.dtype)
        scale = _safe_scale(self.scale).to(device=x.device, dtype=x.dtype)
        return x * scale + center


def fit_channel_stats(x: torch.Tensor, mode: NormMode = "zscore") -> ChannelStats:
    flat = x.reshape(-1, x.shape[-1]).float()
    flat = flat[torch.isfinite(flat).all(dim=-1)]
    if flat.numel() == 0:
        raise ValueError("Cannot fit normalizer on empty/NaN-only tensor.")
    if mode == "none":
        return ChannelStats(mode, torch.zeros(flat.shape[-1]), torch.ones(flat.shape[-1]))
    if mode == "zscore":
        return ChannelStats(mode, flat.mean(dim=0), _safe_scale(flat.std(dim=0)))
    if mode == "robust":
        q25 = flat.quantile(0.25, dim=0)
        q50 = flat.quantile(0.50, dim=0)
        q75 = flat.quantile(0.75, dim=0)
        return ChannelStats(mode, q50, _safe_scale(q75 - q25))
    if mode == "minmax":
        mn = flat.min(dim=0).values
        mx = flat.max(dim=0).values
        return ChannelStats(mode, mn, _safe_scale(mx - mn), min_value=mn, max_value=mx)
    raise ValueError(f"Unsupported normalization mode: {mode}")


@dataclass
class NormalizerBundle:
    wheel: ChannelStats
    gps: ChannelStats
    imu: ChannelStats
    target: ChannelStats

    def to_dict(self) -> dict:
        def pack(stats: ChannelStats) -> dict:
            return {
                "mode": stats.mode,
                "center": stats.center.tolist(),
                "scale": stats.scale.tolist(),
                "min_value": None if stats.min_value is None else stats.min_value.tolist(),
                "max_value": None if stats.max_value is None else stats.max_value.tolist(),
            }
        return {name: pack(getattr(self, name)) for name in ("wheel", "gps", "imu", "target")}

    @staticmethod
    def from_dict(payload: dict) -> "NormalizerBundle":
        def unpack(item: dict) -> ChannelStats:
            min_value = item.get("min_value")
            max_value = item.get("max_value")
            return ChannelStats(
                mode=item["mode"],
                center=torch.tensor(item["center"], dtype=torch.float32),
                scale=torch.tensor(item["scale"], dtype=torch.float32),
                min_value=None if min_value is None else torch.tensor(min_value, dtype=torch.float32),
                max_value=None if max_value is None else torch.tensor(max_value, dtype=torch.float32),
            )
        return NormalizerBundle(
            wheel=unpack(payload["wheel"]),
            gps=unpack(payload["gps"]),
            imu=unpack(payload["imu"]),
            target=unpack(payload["target"]),
        )
