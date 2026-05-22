"""
Causal forward-only Mamba navigation models.
No reverse-time attention anywhere - eliminates train/test GPS lookahead leak.
"""

import torch
import torch.nn as nn
from mamba_ssm import Mamba


def _module_device(module):
    return next(module.parameters()).device


def _to_module_device(module, *tensors):
    device = _module_device(module)
    return tuple(t.to(device) if t.device != device else t for t in tensors)


def _kinematic_delta(wheel_feat, wheel_scale=None, heading_bias=None):
    """
    Deterministic 1 Hz wheel/heading prior.

    wheel_feat[..., 2] is center speed, wheel_feat[..., 4:6] are sin/cos
    heading, and wheel_feat[..., 6] is yaw rate. The learned Mamba heads
    predict residuals on top of this causal dead-reckoning step.
    """
    vc = wheel_feat[..., 2:3]
    sin_theta = wheel_feat[..., 4:5]
    cos_theta = wheel_feat[..., 5:6]
    omega = wheel_feat[..., 6:7]
    if heading_bias is not None:
        sin_b = torch.sin(heading_bias)
        cos_b = torch.cos(heading_bias)
        sin_theta, cos_theta = (
            sin_theta * cos_b + cos_theta * sin_b,
            cos_theta * cos_b - sin_theta * sin_b,
        )
    if wheel_scale is not None:
        vc = vc * wheel_scale.clamp(0.1, 3.0)
    dx = vc * cos_theta
    dy = vc * sin_theta
    zeros = torch.zeros_like(dx)
    return torch.cat([dx, dy, zeros, zeros, omega, zeros], dim=-1)


class CausalMambaBlock(nn.Module):
    """
    Single causal (forward-only) Mamba block with residual + LayerNorm.
    Used everywhere - dead reckoning, GPS, IMU, fusion.
    No reverse-time variant exists in this codebase.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.mamba(x))


class CausalMambaEncoder(nn.Module):
    """Stack of N causal Mamba blocks."""

    def __init__(self, d_model, d_state=16, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList(
            [CausalMambaBlock(d_model, d_state) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CausalMambaNavigator(nn.Module):
    """
    Pure causal forward-only Mamba navigator.

    Three input branches (all causal):
      - Dead reckoning: wheel odometry + heading
      - GPS signal:     relative GPS position + validity flag
      - Heading:        IMU theta, omega, sin, cos

    GPS-validity gated fusion learns WHEN to trust GPS vs dead reckoning.
    Output head predicts delta state [dx, dy, dvx, dvy, dtheta, domega].
    States are integrated externally in the trainer.

    Key design choices vs failed reverse-time version:
      - ALL Mamba blocks are causal (no future leakage)
      - GPS branch sees only past GPS readings
      - d_model=96 matches the final packaged Table II checkpoints
      - Output heads initialized near-zero (start from odometry baseline)
    """

    def __init__(self, d_model=96, d_state=16, n_layers=2,
                 max_pos_residual=8.0):
        super().__init__()
        self.max_pos_residual = float(max_pos_residual)
        self.wheel_scale = nn.Parameter(torch.tensor(1.0))
        self.heading_bias = nn.Parameter(torch.tensor(0.0))

        # Branch 1: Dead reckoning (wheel + IMU)
        # Input: [vl, vr, vc, vdiff, sin(theta), cos(theta), omega] = 7
        self.dr_proj = nn.Linear(7, d_model)
        self.dr_enc = CausalMambaEncoder(d_model, d_state, n_layers)

        # Branch 2: GPS correction
        # Input: [gps_dx_rel, gps_dy_rel, gps_valid] = 3
        # gps_dx/dy are relative to current integrated position
        self.gps_proj = nn.Linear(3, d_model)
        self.gps_enc = CausalMambaEncoder(d_model, d_state, n_layers)

        # Branch 3: IMU heading
        # Input: [theta, omega, sin(theta), cos(theta)] = 4
        self.imu_proj = nn.Linear(4, d_model)
        self.imu_enc = CausalMambaEncoder(d_model, d_state, n_layers=1)

        # GPS-validity gated fusion (+1 for scalar gps_valid flag)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3 + 1, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )
        self.fusion_enc = CausalMambaEncoder(d_model, d_state, n_layers=1)
        self.gps_conf_head = nn.Sequential(
            nn.Linear(d_model + 3 + 7 + 4 + 1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Separate output heads: position (most critical) vs velocity/heading
        self.pos_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 2),
        )
        self.state_head = nn.Linear(d_model, 4)

        # Learnable per-output log uncertainty (heteroscedastic loss)
        self.log_sigma = nn.Parameter(torch.zeros(6))

        self._init_output_heads()

    def _init_output_heads(self):
        """Initialize output heads near zero - starts from odometry baseline."""
        for module in [self.pos_head, self.state_head, self.gps_conf_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        final = self.gps_conf_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.constant_(final.bias, -2.0)

    def forward(
        self,
        wheel_feat,
        gps_feat,
        imu_feat,
        gps_valid,
        return_confidence=False,
    ):
        """
        wheel_feat : (B, T, 7)
        gps_feat   : (B, T, 3)  GPS relative to current integrated pos
        imu_feat   : (B, T, 4)
        gps_valid  : (B, T, 1)  float 0.0 or 1.0
        Returns:
            delta     : (B, T, 6)
            log_sigma : (6,)
        """
        wheel_feat, gps_feat, imu_feat, gps_valid = _to_module_device(
            self, wheel_feat, gps_feat, imu_feat, gps_valid
        )
        dr = self.dr_enc(self.dr_proj(wheel_feat))
        gps = self.gps_enc(self.gps_proj(gps_feat))
        imu = self.imu_enc(self.imu_proj(imu_feat))

        fused = self.fusion(torch.cat([dr, gps, imu, gps_valid], dim=-1))
        fused = self.fusion_enc(fused)

        base_delta = _kinematic_delta(
            wheel_feat, self.wheel_scale, self.heading_bias
        )
        delta_pos = (
            base_delta[..., :2]
            + self.max_pos_residual * torch.tanh(self.pos_head(fused))
        )
        delta_other = base_delta[..., 2:] + 0.3 * torch.tanh(
            self.state_head(fused)
        )
        delta = torch.cat([delta_pos, delta_other], dim=-1)
        delta = delta + 0.0 * self.log_sigma.view(1, 1, -1)

        if return_confidence:
            conf_in = torch.cat(
                [fused, gps_feat, wheel_feat, imu_feat, gps_valid], dim=-1
            )
            gps_conf = torch.sigmoid(self.gps_conf_head(conf_in)) * gps_valid
            return delta, self.log_sigma, gps_conf

        return delta, self.log_sigma


class SplitCausalMambaNavigator(nn.Module):
    """
    Split variant: two independent Mamba stacks, one per state group.
    Mirrors Split-KalmanNet's design of separate networks for different
    parts of the state.

    Stack A: position + velocity [px, py, vx, vy]
    Stack B: heading             [theta, omega]

    Both stacks are fully causal. No shared weights between stacks.
    """

    def __init__(self, d_model=96, d_state=16, n_layers=2,
                 max_pos_residual=8.0):
        super().__init__()
        self.max_pos_residual = float(max_pos_residual)
        self.wheel_scale = nn.Parameter(torch.tensor(1.0))
        self.heading_bias = nn.Parameter(torch.tensor(0.0))

        # Stack A: position/velocity
        self.a_dr_proj = nn.Linear(7, d_model)
        self.a_dr_enc = CausalMambaEncoder(d_model, d_state, n_layers)
        self.a_gps_proj = nn.Linear(3, d_model)
        self.a_gps_enc = CausalMambaEncoder(d_model, d_state, n_layers)
        self.a_fusion = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.a_post_enc = CausalMambaEncoder(d_model, d_state, n_layers=1)
        self.gps_conf_head = nn.Sequential(
            nn.Linear(d_model + 3 + 7 + 1, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.a_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 4),
        )

        # Stack B: heading
        self.b_imu_proj = nn.Linear(4, d_model)
        self.b_imu_enc = CausalMambaEncoder(d_model, d_state, n_layers)
        self.b_dr_proj = nn.Linear(7, d_model)
        self.b_dr_enc = CausalMambaEncoder(d_model, d_state, n_layers=1)
        self.b_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.b_post_enc = CausalMambaEncoder(d_model, d_state, n_layers=1)
        self.b_head = nn.Linear(d_model, 2)

        self.log_sigma = nn.Parameter(torch.zeros(6))
        self._init_heads()

    def _init_heads(self):
        for module in [self.a_head, self.b_head, self.gps_conf_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        final = self.gps_conf_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.constant_(final.bias, -2.0)

    def forward(
        self,
        wheel_feat,
        gps_feat,
        imu_feat,
        gps_valid,
        return_confidence=False,
    ):
        wheel_feat, gps_feat, imu_feat, gps_valid = _to_module_device(
            self, wheel_feat, gps_feat, imu_feat, gps_valid
        )
        a_dr = self.a_dr_enc(self.a_dr_proj(wheel_feat))
        a_gps = self.a_gps_enc(self.a_gps_proj(gps_feat))
        a_out = self.a_fusion(torch.cat([a_dr, a_gps, gps_valid], dim=-1))
        a_out = self.a_post_enc(a_out)
        base_delta = _kinematic_delta(
            wheel_feat, self.wheel_scale, self.heading_bias
        )
        a_raw = self.a_head(a_out)
        delta_pos = (
            base_delta[..., :2]
            + self.max_pos_residual * torch.tanh(a_raw[..., :2])
        )
        delta_pos_vel = torch.cat([delta_pos, a_raw[..., 2:4]], dim=-1)

        b_imu = self.b_imu_enc(self.b_imu_proj(imu_feat))
        b_dr = self.b_dr_enc(self.b_dr_proj(wheel_feat))
        b_out = self.b_fusion(torch.cat([b_imu, b_dr], dim=-1))
        b_out = self.b_post_enc(b_out)
        delta_heading = base_delta[..., 4:6] + self.b_head(b_out)

        delta = torch.cat([delta_pos_vel, delta_heading], dim=-1)
        delta = delta + 0.0 * self.log_sigma.view(1, 1, -1)
        if return_confidence:
            conf_in = torch.cat([a_out, gps_feat, wheel_feat, gps_valid], dim=-1)
            gps_conf = torch.sigmoid(self.gps_conf_head(conf_in)) * gps_valid
            return delta, self.log_sigma, gps_conf
        return delta, self.log_sigma
