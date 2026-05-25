import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except Exception:
    Mamba = None


D_WHEEL = 7
D_GPS = 3
D_IMU = 4
D_STATE = 6
D_EMBED = 64
D_MAMBA = 128
N_MAMBA = 4
D_STATE_SSM = 32
D_CONV = 4
EXPAND = 2


class SensorEmbedding(nn.Module):
    """
    Projects wheel/GPS/IMU into a shared token space and applies a learned,
    differentiable GPS-validity gate to the GPS token.
    """

    def __init__(self, d_embed=D_EMBED):
        super().__init__()
        self.wheel_proj = nn.Linear(D_WHEEL, d_embed)
        self.gps_proj = nn.Linear(D_GPS, d_embed)
        self.imu_proj = nn.Linear(D_IMU, d_embed)
        self.gps_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self.modality_tokens = nn.Parameter(torch.randn(3, d_embed) * 0.02)

    def forward(self, wheel, gps, imu):
        w_emb = self.wheel_proj(wheel)
        g_emb = self.gps_proj(gps)
        i_emb = self.imu_proj(imu)

        gps_valid = gps[:, :, 2:3]
        g_emb = g_emb * self.gps_gate(gps_valid)

        w_emb = w_emb + self.modality_tokens[0]
        g_emb = g_emb + self.modality_tokens[1]
        i_emb = i_emb + self.modality_tokens[2]
        return torch.stack([w_emb, g_emb, i_emb], dim=2)


class CausalRNNBlock(nn.Module):
    """Portable causal fallback used when mamba-ssm cannot be imported."""

    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.gru = nn.GRU(d_model, d_model, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(self.norm(x))
        return x + out


class CausalMambaBlock(nn.Module):
    """Single forward-only Mamba SSM block with pre-norm and residual."""

    def __init__(
        self,
        d_model=D_MAMBA,
        d_state=D_STATE_SSM,
        d_conv=D_CONV,
        expand=EXPAND,
        use_mamba=True,
    ):
        super().__init__()
        if Mamba is None or not use_mamba:
            self.block = CausalRNNBlock(d_model)
        else:
            self.norm = nn.RMSNorm(d_model)
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.block = None

    def forward(self, x):
        if self.block is not None:
            return self.block(x)
        return x + self.mamba(self.norm(x))


class UnifiedMambaEncoder(nn.Module):
    """
    Encodes the interleaved causal sequence:
    [W_t, G_t, I_t, W_{t+1}, G_{t+1}, I_{t+1}, ...].
    """

    def __init__(
        self,
        d_embed=D_EMBED,
        d_model=D_MAMBA,
        n_layers=N_MAMBA,
        d_state=D_STATE_SSM,
        d_conv=D_CONV,
        expand=EXPAND,
        use_mamba=True,
    ):
        super().__init__()
        self.input_proj = nn.Linear(d_embed, d_model)
        self.layers = nn.ModuleList(
            [
                CausalMambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_mamba=use_mamba,
                )
                for _ in range(n_layers)
            ]
        )
        self.output_proj = nn.Linear(3 * d_model, d_model)
        self.output_norm = nn.RMSNorm(d_model)

    def forward(self, tokens):
        bsz, steps, sensors, dim = tokens.shape
        x = tokens.reshape(bsz, steps * sensors, dim)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(bsz, steps, sensors * x.shape[-1])
        return self.output_norm(self.output_proj(x))


class ResidualPredictionHead(nn.Module):
    """Predicts state delta mean and per-dimension log variance."""

    def __init__(self, d_model=D_MAMBA, d_state=D_STATE):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )
        self.mean_head = nn.Linear(d_model, d_state)
        self.logvar_head = nn.Linear(d_model, d_state)

    def forward(self, x):
        h = self.shared(x)
        return self.mean_head(h), self.logvar_head(h)


def wrap_to_pi(angle):
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def kinematic_integrate(x_prev, delta, wheel_feat):
    """
    Integrates local-frame position deltas into the global frame using the
    current wheel/IMU heading features.
    """

    sin_theta = wheel_feat[:, 4:5]
    cos_theta = wheel_feat[:, 5:6]
    dx = delta[:, 0:1]
    dy = delta[:, 1:2]

    dx_global = dx * cos_theta - dy * sin_theta
    dy_global = dx * sin_theta + dy * cos_theta
    theta = wrap_to_pi(x_prev[:, 4:5] + delta[:, 4:5])

    return torch.cat(
        [
            x_prev[:, 0:1] + dx_global,
            x_prev[:, 1:2] + dy_global,
            x_prev[:, 2:3] + delta[:, 2:3],
            x_prev[:, 3:4] + delta[:, 3:4],
            theta,
            delta[:, 5:6],
        ],
        dim=-1,
    )


class SlimMambaNav(nn.Module):
    """
    Single-stream causal Mamba navigation model for wheel/GPS/IMU fusion.
    """

    def __init__(
        self,
        d_embed=D_EMBED,
        d_model=D_MAMBA,
        n_layers=N_MAMBA,
        d_state_ssm=D_STATE_SSM,
        d_conv=D_CONV,
        expand=EXPAND,
        use_mamba=True,
    ):
        super().__init__()
        self.embedder = SensorEmbedding(d_embed=d_embed)
        self.encoder = UnifiedMambaEncoder(
            d_embed=d_embed,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state_ssm,
            d_conv=d_conv,
            expand=expand,
            use_mamba=use_mamba,
        )
        self.head = ResidualPredictionHead(d_model=d_model)

    @classmethod
    def from_config(cls, cfg):
        return cls(
            d_embed=cfg.d_embed,
            d_model=cfg.d_mamba,
            n_layers=cfg.n_mamba,
            d_state_ssm=cfg.d_state_ssm,
            d_conv=cfg.d_conv,
            expand=cfg.expand,
            use_mamba=cfg.use_mamba,
        )

    def forward(self, wheel, gps, imu, x0):
        _, steps, _ = wheel.shape
        tokens = self.embedder(wheel, gps, imu)
        features = self.encoder(tokens)
        delta_mean, delta_logvar = self.head(features)

        states = []
        x_t = x0
        for t in range(steps):
            x_t = kinematic_integrate(x_t, delta_mean[:, t], wheel[:, t])
            states.append(x_t)
        states = torch.stack(states, dim=1)
        return states, delta_mean, delta_logvar.clamp(min=-10.0, max=10.0)

    @torch.no_grad()
    def predict_trajectory(self, wheel, gps, imu, x0):
        states, _, _ = self.forward(wheel, gps, imu, x0)
        return states
