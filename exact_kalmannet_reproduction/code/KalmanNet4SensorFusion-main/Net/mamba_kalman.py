"""Forward-only Mamba replacements for KalmanNet gain networks."""

from __future__ import annotations

import typing
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as func

try:
    from mamba_ssm import Mamba
except ImportError as exc:  # pragma: no cover - environment/dependency guard.
    raise ImportError("mamba-ssm not installed. Run: pip install mamba-ssm") from exc

from Net.models.fusion_net.fusion_kalman_net import FusionKalmanNet, FusionSplitKalmanNet
from Net.utils import MODELS


class _MambaGainCore(nn.Module):
    """Causal Mamba feature-to-gain network with optional step cache."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.d_model = int(d_model)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.mamba_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_dim),
        )
        final = self.output_proj[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self._cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self._cache = []
        for layer in self.mamba_layers:
            conv_state, ssm_state = layer.allocate_inference_cache(batch_size, 1, dtype=dtype)
            self._cache.append((conv_state.to(device), ssm_state.to(device)))

    def detach_state(self) -> None:
        if self._cache is None:
            return
        self._cache = [(conv.detach(), ssm.detach()) for conv, ssm in self._cache]

    def _run_sequence(self, x: torch.Tensor) -> torch.Tensor:
        for mamba, norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = mamba(norm(x)) + residual
        return x

    def _run_step(self, x: torch.Tensor) -> torch.Tensor:
        if self._cache is None or len(self._cache) != len(self.mamba_layers):
            self.init_state(x.shape[0], x.device, x.dtype)
        assert self._cache is not None
        x = x.unsqueeze(1)
        next_cache = []
        for idx, (mamba, norm) in enumerate(zip(self.mamba_layers, self.layer_norms)):
            residual = x
            conv_state, ssm_state = self._cache[idx]
            y, conv_state, ssm_state = mamba.step(norm(x), conv_state, ssm_state)
            x = y + residual
            next_cache.append((conv_state, ssm_state))
        self._cache = next_cache
        return x.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        if x.dim() == 2:
            x = self._run_step(x)
        elif x.dim() == 3:
            x = self._run_sequence(x)
        else:
            raise ValueError(f"Expected features with 2 or 3 dims, got {tuple(x.shape)}")
        return self.output_proj(x)


class _FeatureMixin:
    gps_gate: nn.Parameter

    @staticmethod
    def _feature_dict_to_tensor(features_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        f1 = features_dict["F1"]
        f2 = features_dict["F2"]
        f3 = features_dict["F3"]
        f4 = features_dict["F4"]
        gps = features_dict.get("gps_available", torch.ones_like(f3[..., :1]))
        f3_gated = f3 * gps
        f4_gated = f4 * gps
        return torch.cat([f1, f2, f3_gated, f4_gated], dim=-1)


@MODELS.register_module()
class MambaKalmanNet(FusionKalmanNet, _FeatureMixin):
    """Drop-in replacement for KalmanNet's GRU with forward-only Mamba SSM."""

    def __init__(
        self,
        input_dim: int | None = None,
        output_dim: int | None = None,
        d_model: int = 64,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        params: Dict[str, Any] | Any | None = None,
        in_mult_KNet: int = 5,
        out_mult_KNet: int = 40,
        device: str | None = None,
    ) -> None:
        self.mamba_d_model = d_model
        self.mamba_n_layers = n_layers
        self.mamba_d_state = d_state
        self.mamba_d_conv = d_conv
        self.mamba_expand = expand
        self._standalone = params is None
        if self._standalone:
            if input_dim is None or output_dim is None:
                raise ValueError("Standalone MambaKalmanNet requires input_dim and output_dim")
            nn.Module.__init__(self)
            self.gain_core = _MambaGainCore(
                input_dim=input_dim,
                output_dim=output_dim,
                d_model=d_model,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.gps_gate = nn.Parameter(torch.ones(1))
        else:
            super().__init__(
                params=params,
                in_mult_KNet=in_mult_KNet,
                out_mult_KNet=out_mult_KNet,
                device=device,
            )

    def init_KGainNet(
        self,
        prior_Q: torch.Tensor | None = None,
        prior_Sigma: torch.Tensor | None = None,
        prior_S: torch.Tensor | None = None,
    ) -> None:
        super().init_KGainNet(prior_Q=prior_Q, prior_Sigma=prior_Sigma, prior_S=prior_S)
        self.mamba_input_dim = 2 * self.state_dim + 2 * self.observation_dim
        self.mamba_output_dim = self.state_dim * self.observation_dim
        self.gain_core = _MambaGainCore(
            input_dim=self.mamba_input_dim,
            output_dim=self.mamba_output_dim,
            d_model=self.mamba_d_model,
            n_layers=self.mamba_n_layers,
            d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv,
            expand=self.mamba_expand,
        )
        self.gps_gate = nn.Parameter(torch.ones(1))

    def init_beliefs(self, state: torch.Tensor) -> None:
        super().init_beliefs(state)
        self.gain_core.init_state(state.shape[0], state.device, state.dtype)

    @typing.no_type_check
    def _forward_(self, sensor: torch.Tensor, correction: torch.Tensor | None) -> torch.Tensor:
        m1x_prior, f_jac, m1y, h_jac = self.step_prior(self.m1x_posterior, sensor)
        mask = torch.isnan(correction)
        self._gps_available = torch.isfinite(correction).all(dim=1).float()
        y = torch.zeros_like(self.y_previous)
        y[mask] = m1y[mask]
        y[~mask] = correction[~mask]

        obs_diff, obs_innov_diff, fw_evol_diff, fw_update_diff = self.step_KGain_est(
            y,
            m1y,
            self.y_previous,
            self.m1x_posterior,
            self.m1x_posterior_previous,
            self.m1x_prior_previous,
        )
        kalman_gain = self.KGain_step(
            obs_diff,
            obs_innov_diff,
            fw_evol_diff,
            fw_update_diff,
        ).reshape(self.batch_size, self.state_dim, self.observation_dim)

        innov = y - m1y
        m1x_posterior = torch.baddbmm(m1x_prior, kalman_gain, innov)
        m1x_posterior = self.params.convert(m1x_posterior)
        bad_posterior = ~torch.isfinite(m1x_posterior).flatten(1).all(dim=1)
        if torch.any(bad_posterior):
            m1x_posterior[bad_posterior] = torch.nan_to_num(
                m1x_prior.detach().clone()[bad_posterior],
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )
        self.innovation = innov.detach().clone()
        self.m1x_posterior_previous = self.m1x_posterior.detach().clone()
        self.m1x_posterior = m1x_posterior.detach().clone()
        self.m1x_prior_previous = m1x_prior.detach().clone()
        self.y_previous = y.detach().clone()
        return m1x_posterior.clone()

    def KGain_step(
        self,
        obs_diff: torch.Tensor,
        obs_innov_diff: torch.Tensor,
        fw_evol_diff: torch.Tensor,
        fw_update_diff: torch.Tensor,
    ) -> torch.Tensor:
        features = {
            "F1": fw_evol_diff,
            "F2": fw_update_diff,
            "F3": obs_diff,
            "F4": obs_innov_diff,
            "gps_available": getattr(
                self,
                "_gps_available",
                torch.ones(obs_diff.shape[0], 1, device=obs_diff.device, dtype=obs_diff.dtype),
            )
            * self.gps_gate.clamp_min(0.0),
        }
        return self.gain_core(self._feature_dict_to_tensor(features)).unsqueeze(0)

    def forward(self, *args, **kwargs):
        if self._standalone:
            features_dict = args[0] if args else kwargs["features_dict"]
            return self.gain_core(self._feature_dict_to_tensor(features_dict))
        return super().forward(*args, **kwargs)

    def _detach(self) -> None:
        if self._standalone:
            self.gain_core.detach_state()
            return
        super()._detach()
        self.gain_core.detach_state()


@MODELS.register_module()
class MambaSplitKalmanNet(FusionSplitKalmanNet, _FeatureMixin):
    """Drop-in replacement for Split-KalmanNet using two forward-only Mamba SSMs."""

    def __init__(
        self,
        input_dim: int | None = None,
        p_dim: int | None = None,
        s_dim: int | None = None,
        d_model: int = 64,
        n_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        params: Dict[str, Any] | Any | None = None,
        gru_scale_s: int = 2,
        nGRU: int = 2,
        device: str | None = None,
    ) -> None:
        self.mamba_d_model = d_model
        self.mamba_n_layers = n_layers
        self.mamba_d_state = d_state
        self.mamba_d_conv = d_conv
        self.mamba_expand = expand
        self._standalone = params is None
        if self._standalone:
            if input_dim is None or p_dim is None or s_dim is None:
                raise ValueError("Standalone MambaSplitKalmanNet requires input_dim, p_dim, and s_dim")
            nn.Module.__init__(self)
            self.mamba_P = MambaKalmanNet(
                input_dim=input_dim,
                output_dim=p_dim,
                d_model=d_model,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.mamba_S = MambaKalmanNet(
                input_dim=input_dim,
                output_dim=s_dim,
                d_model=d_model,
                n_layers=n_layers,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            super().__init__(params=params, gru_scale_s=gru_scale_s, nGRU=nGRU, device=device)

    def _init_net(self) -> None:
        self.mamba_input_dim = 2 * self.state_dim + 2 * self.observation_dim
        self.output_dim_1 = self.state_dim * self.state_dim
        self.output_dim_2 = self.observation_dim**2
        self.mamba_P = _MambaGainCore(
            input_dim=self.mamba_input_dim,
            output_dim=self.output_dim_1,
            d_model=self.mamba_d_model,
            n_layers=self.mamba_n_layers,
            d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv,
            expand=self.mamba_expand,
        )
        self.mamba_S = _MambaGainCore(
            input_dim=self.mamba_input_dim,
            output_dim=self.output_dim_2,
            d_model=self.mamba_d_model,
            n_layers=self.mamba_n_layers,
            d_state=self.mamba_d_state,
            d_conv=self.mamba_d_conv,
            expand=self.mamba_expand,
        )
        self.gps_gate = nn.Parameter(torch.ones(1))

    def init_beliefs(self, state: torch.Tensor) -> None:
        super().init_beliefs(state)
        self.mamba_P.init_state(state.shape[0], state.device, state.dtype)
        self.mamba_S.init_state(state.shape[0], state.device, state.dtype)

    def forward(self, *args, **kwargs):
        if self._standalone:
            features_dict = args[0] if args else kwargs["features_dict"]
            return self.mamba_P(features_dict), self.mamba_S(features_dict)
        return super().forward(*args, **kwargs)

    def _forward_(self, state_inno, observation_inno, diff_state, diff_obs, linear_error, h_jac, f_jac):
        gps_available = getattr(
            self,
            "_gps_available",
            torch.ones(diff_obs.shape[0], 1, 1, device=diff_obs.device, dtype=diff_obs.dtype),
        ).squeeze(-1)
        features = {
            "F1": torch.squeeze(diff_state, -1),
            "F2": torch.squeeze(state_inno, -1),
            "F3": torch.squeeze(diff_obs, -1),
            "F4": torch.squeeze(observation_inno, -1),
            "gps_available": gps_available * self.gps_gate.clamp_min(0.0),
        }
        x = self._feature_dict_to_tensor(features)
        batch_size = state_inno.shape[0]
        p_flat = self.mamba_P(x)
        s_flat = self.mamba_S(x)
        Pk = p_flat.reshape(batch_size, self.state_dim, self.state_dim)
        Sk = s_flat.reshape(batch_size, self.observation_dim, self.observation_dim)
        return Pk, Sk

    @typing.no_type_check
    def forward(self, sensor: torch.Tensor | Dict[str, torch.Tensor], correction: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore[override]
        if self._standalone:
            return self.mamba_P(sensor), self.mamba_S(sensor)
        if correction is None:
            raise ValueError("Drop-in MambaSplitKalmanNet requires correction tensor")
        m1x_prior, f_jac, m1y, h_jac = self.predict(self.m1x_posterior, sensor)
        mask = torch.isnan(correction)
        self._gps_available = torch.isfinite(correction).all(dim=1).float()
        y = torch.zeros_like(self.y_previous)
        y[mask] = m1y[mask]
        y[~mask] = correction[~mask]

        state_inno = self.m1x_posterior_previous - self.m1x_prior_previous
        innov = y - m1y
        diff_state = self.m1x_posterior - self.m1x_posterior_previous
        diff_obs = y - self.y_previous
        linear_error = m1y - h_jac @ m1x_prior

        Pk, Sk = self._forward_(state_inno, innov, diff_state, diff_obs, linear_error, h_jac, f_jac)
        kalman_gain = Pk @ torch.transpose(h_jac, -1, -2) @ Sk
        m1x_posterior = torch.baddbmm(m1x_prior, kalman_gain, innov)
        m1x_posterior = self.params.convert(m1x_posterior)
        bad_posterior = ~torch.isfinite(m1x_posterior).flatten(1).all(dim=1)
        if torch.any(bad_posterior):
            m1x_posterior[bad_posterior] = torch.nan_to_num(
                m1x_prior.detach().clone()[bad_posterior],
                nan=0.0,
                posinf=1e6,
                neginf=-1e6,
            )
        self.innovation = innov
        self.m1x_posterior_previous = self.m1x_posterior.detach().clone()
        self.m1x_posterior = m1x_posterior.detach().clone()
        self.m1x_prior_previous = m1x_prior.detach().clone()
        self.y_previous = y.detach().clone()
        return m1x_posterior.clone()

    def _detach(self) -> None:
        if self._standalone:
            self.mamba_P._detach()
            self.mamba_S._detach()
            return
        super()._detach()
        self.mamba_P.detach_state()
        self.mamba_S.detach_state()
