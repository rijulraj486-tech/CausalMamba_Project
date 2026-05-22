"""Losses for the pure Mamba neural filter."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PureMambaFilterLossConfig:
    position_weight: float = 1.0
    velocity_weight: float = 0.2
    acceleration_weight: float = 0.05
    heading_weight: float = 0.05
    smoothness_weight: float = 0.01
    trajectory_consistency_weight: float = 0.0
    nll_weight: float = 0.0
    dt: float = 1.0


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def pure_mamba_filter_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    cfg: PureMambaFilterLossConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    pred = torch.nan_to_num(output["state_estimates"].float(), nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)

    pos = F.mse_loss(pred[..., :2], target[..., :2])
    vel = F.mse_loss(pred[..., 2:4], target[..., 2:4])
    acc = F.mse_loss(pred[..., 4:6], target[..., 4:6])
    heading = F.mse_loss(wrap_to_pi(pred[..., 6] - target[..., 6]), torch.zeros_like(target[..., 6]))

    if pred.shape[1] > 1:
        pred_delta_xy = pred[:, 1:, :2] - pred[:, :-1, :2]
        target_delta_xy = target[:, 1:, :2] - target[:, :-1, :2]
        smooth = F.mse_loss(pred_delta_xy[:, 1:], pred_delta_xy[:, :-1]) if pred_delta_xy.shape[1] > 1 else _zero_like_loss(pred)
        traj = F.mse_loss(pred_delta_xy, target_delta_xy)
    else:
        smooth = _zero_like_loss(pred)
        traj = _zero_like_loss(pred)

    nll = _zero_like_loss(pred)
    if "log_var" in output and cfg.nll_weight > 0.0:
        log_var = torch.nan_to_num(output["log_var"].float(), nan=0.0, posinf=6.0, neginf=-8.0).clamp(-8.0, 6.0)
        inv_var = torch.exp(-log_var)
        nll = 0.5 * (((pred[..., :2] - target[..., :2]) ** 2) * inv_var + log_var).mean()

    loss = (
        cfg.position_weight * pos
        + cfg.velocity_weight * vel
        + cfg.acceleration_weight * acc
        + cfg.heading_weight * heading
        + cfg.smoothness_weight * smooth
        + cfg.trajectory_consistency_weight * traj
        + cfg.nll_weight * nll
    )
    parts = {
        "loss": float(loss.detach().item()),
        "position": float(pos.detach().item()),
        "velocity": float(vel.detach().item()),
        "acceleration": float(acc.detach().item()),
        "heading": float(heading.detach().item()),
        "smoothness": float(smooth.detach().item()),
        "trajectory_consistency": float(traj.detach().item()),
        "nll": float(nll.detach().item()),
    }
    return loss, parts
