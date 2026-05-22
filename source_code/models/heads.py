"""Physics-aware state heads and differentiable trajectory decoding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


STATE_NAMES = ("x", "y", "vx", "vy", "ax", "ay", "heading", "yaw_rate", "speed")
STATE_DIM = len(STATE_NAMES)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PhysicsStateHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, probabilistic: bool = True, dropout: float = 0.05) -> None:
        super().__init__()
        self.probabilistic = probabilistic
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, STATE_DIM + (2 if probabilistic else 0)),
        )
        with torch.no_grad():
            final = self.net[-1]
            assert isinstance(final, nn.Linear)
            final.weight.mul_(0.02)
            final.bias.zero_()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(x)
        state_raw = raw[..., :STATE_DIM]
        state = torch.cat(
            [
                state_raw[..., 0:6],
                wrap_to_pi(state_raw[..., 6:7]),
                state_raw[..., 7:8],
                torch.nn.functional.softplus(state_raw[..., 8:9]),
            ],
            dim=-1,
        )
        out = {"state_delta": state}
        if self.probabilistic:
            out["log_var"] = torch.clamp(raw[..., STATE_DIM:STATE_DIM + 2], min=-6.0, max=5.0)
        return out


class TrajectoryDecoder(nn.Module):
    """Converts state deltas into an XY trajectory with differentiable integration."""

    def __init__(
        self,
        dt: float = 1.0,
        mode: str = "integrated",
        pos_residual_scale: float = 80.0,
        vel_scale: float = 5.0,
        acc_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if mode not in {"integrated", "direct", "hybrid"}:
            raise ValueError("decoder mode must be integrated, direct, or hybrid")
        self.dt = float(dt)
        self.mode = mode
        self.pos_residual_scale = float(pos_residual_scale)
        self.vel_scale = float(vel_scale)
        self.acc_scale = float(acc_scale)

    def forward(
        self,
        state_delta: torch.Tensor,
        initial_state: torch.Tensor,
        anchor_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = state_delta.shape
        pos = initial_state[:, :2]
        if initial_state.shape[-1] >= 6:
            vel = initial_state[:, 2:4]
            heading_prev = initial_state[:, 4]
            yaw_prev = initial_state[:, 5]
        else:
            vel = state_delta.new_zeros(bsz, 2)
            heading_prev = state_delta.new_zeros(bsz)
            yaw_prev = state_delta.new_zeros(bsz)
        decoded = []

        for t in range(seq_len):
            raw = state_delta[:, t, :]
            if anchor_position is None:
                learned_pos = initial_state[:, :2] + self.pos_residual_scale * torch.tanh(raw[:, :2])
            else:
                learned_pos = anchor_position[:, t, :] + self.pos_residual_scale * torch.tanh(raw[:, :2])
            axay = self.acc_scale * torch.tanh(raw[:, 4:6])
            vxvy = self.vel_scale * torch.tanh(raw[:, 2:4])
            heading = wrap_to_pi(raw[:, 6])
            yaw_rate = 0.5 * torch.tanh(raw[:, 7])
            speed = torch.nn.functional.softplus(raw[:, 8])

            if self.mode == "direct":
                pos_next = learned_pos
                vel_next = vxvy
            else:
                vel_next = 0.65 * vel + 0.35 * vxvy + axay * self.dt
                integrated_pos = pos + vel_next * self.dt + 0.5 * axay * (self.dt ** 2)
                if self.mode == "hybrid":
                    blend = 0.65 if anchor_position is not None else 0.25
                    pos_next = integrated_pos + blend * (learned_pos - integrated_pos)
                else:
                    pos_next = integrated_pos

            state_next = torch.cat(
                [
                    pos_next,
                    vel_next,
                    axay,
                    heading.unsqueeze(-1),
                    yaw_rate.unsqueeze(-1),
                    speed.unsqueeze(-1),
                ],
                dim=-1,
            )
            decoded.append(state_next)
            pos = pos_next
            vel = vel_next
            heading_prev = heading
            yaw_prev = yaw_rate

        return torch.stack(decoded, dim=1)
