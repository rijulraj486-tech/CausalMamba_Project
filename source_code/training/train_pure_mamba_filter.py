#!/usr/bin/env python3
"""Train the pure Mamba neural filter on NCLT Wheels + GPS + IMU."""

from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.nclt_fusion_dataset import (  # noqa: E402
    extract_gps_features,
    extract_imu_features,
    extract_wheel_features,
    load_sessions,
    wrap_to_pi,
)
from datasets.normalization import ChannelStats, NormMode, fit_channel_stats  # noqa: E402
from models.pure_mamba_filter import PureMambaFilterConfig, build_pure_mamba_filter  # noqa: E402
from training.filter_losses import PureMambaFilterLossConfig, pure_mamba_filter_loss  # noqa: E402
from training.optim import OptimConfig, build_optimizer, build_scheduler  # noqa: E402
from utils.checkpoint import EMA, save_checkpoint  # noqa: E402
from utils.config import load_yaml, update_dataclass  # noqa: E402
from utils.seed import set_seed  # noqa: E402


STATE_DIM = 8


@dataclass
class PureMambaFilterDataConfig:
    train_path: str = (
        "radar_longseq_benchmark/Radar_Tracking_Benchmark/paper_kalmannet_bimamba_compare/"
        "external/KalmanNet4SensorFusion-main/data/NCLT/processed/train.pt"
    )
    val_path: str = (
        "radar_longseq_benchmark/Radar_Tracking_Benchmark/paper_kalmannet_bimamba_compare/"
        "external/KalmanNet4SensorFusion-main/data/NCLT/processed/val.pt"
    )
    test_path: str = (
        "radar_longseq_benchmark/Radar_Tracking_Benchmark/paper_kalmannet_bimamba_compare/"
        "external/KalmanNet4SensorFusion-main/data/NCLT/processed/test.pt"
    )
    seq_len: int = 200
    stride: int = 100
    dt: float = 1.0
    sensor_normalization: NormMode = "robust"
    target_xy_normalization: NormMode = "robust"


@dataclass
class PureMambaFilterTrainConfig:
    output_dir: str = "experiments/pure_mamba_filter"
    seed: int = 21
    epochs: int = 80
    batch_size: int = 64
    num_workers: int = 2
    prefetch_factor: int = 2
    grad_clip: float = 1.0
    amp: bool = False
    ema: bool = True
    ema_decay: float = 0.997
    early_stopping_patience: int = 12
    curriculum: tuple[int, ...] = (20, 50, 100, 200)
    curriculum_epochs: tuple[int, ...] = (10, 15, 20, 35)


@dataclass
class FilterNormalizerBundle:
    wheel: ChannelStats
    gps: ChannelStats
    imu: ChannelStats
    target_xy: ChannelStats

    def normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        normalized = target.clone().float()
        normalized[..., :2] = self.target_xy.normalize(normalized[..., :2])
        return normalized

    def denormalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        return self.target_xy.denormalize(xy.float())

    def normalize_initial_state(self, initial_state: torch.Tensor) -> torch.Tensor:
        initial_state = initial_state.clone().float()
        initial_state[..., :2] = self.target_xy.normalize(initial_state[..., :2])
        return torch.nan_to_num(initial_state, nan=0.0, posinf=0.0, neginf=0.0)

    def to_dict(self) -> dict[str, Any]:
        def pack(stats: ChannelStats) -> dict[str, Any]:
            return {
                "mode": stats.mode,
                "center": stats.center.tolist(),
                "scale": stats.scale.tolist(),
                "min_value": None if stats.min_value is None else stats.min_value.tolist(),
                "max_value": None if stats.max_value is None else stats.max_value.tolist(),
            }

        return {
            "wheel": pack(self.wheel),
            "gps": pack(self.gps),
            "imu": pack(self.imu),
            "target_xy": pack(self.target_xy),
        }

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "FilterNormalizerBundle":
        def unpack(item: dict[str, Any]) -> ChannelStats:
            min_value = item.get("min_value")
            max_value = item.get("max_value")
            return ChannelStats(
                mode=item["mode"],
                center=torch.tensor(item["center"], dtype=torch.float32),
                scale=torch.tensor(item["scale"], dtype=torch.float32),
                min_value=None if min_value is None else torch.tensor(min_value, dtype=torch.float32),
                max_value=None if max_value is None else torch.tensor(max_value, dtype=torch.float32),
            )

        return FilterNormalizerBundle(
            wheel=unpack(payload["wheel"]),
            gps=unpack(payload["gps"]),
            imu=unpack(payload["imu"]),
            target_xy=unpack(payload["target_xy"]),
        )


def derive_filter_targets(item: dict[str, Any], dt: float) -> torch.Tensor:
    pos = item["ground_truth"].float()
    theta = item["theta_gt"].float().reshape(-1)
    vel = torch.zeros_like(pos)
    acc = torch.zeros_like(pos)
    if pos.shape[0] > 1:
        vel[1:] = (pos[1:] - pos[:-1]) / max(dt, 1.0e-6)
        vel[0] = item["initial_state"].float()[2:4]
        acc[1:] = (vel[1:] - vel[:-1]) / max(dt, 1.0e-6)
    yaw_rate = torch.zeros_like(theta)
    if theta.shape[0] > 1:
        yaw_rate[1:] = wrap_to_pi(theta[1:] - theta[:-1]) / max(dt, 1.0e-6)
        yaw_rate[0] = float(item["initial_state"].float()[5])
    return torch.stack(
        [pos[:, 0], pos[:, 1], vel[:, 0], vel[:, 1], acc[:, 0], acc[:, 1], theta, yaw_rate],
        dim=-1,
    )


def fit_filter_normalizers(sessions: list[dict[str, Any]], cfg: PureMambaFilterDataConfig) -> FilterNormalizerBundle:
    wheel = torch.cat([extract_wheel_features(item, cfg.dt) for item in sessions], dim=0)
    gps = torch.cat([extract_gps_features(item, cfg.dt) for item in sessions], dim=0)
    imu = torch.cat([extract_imu_features(item, cfg.dt) for item in sessions], dim=0)
    target_xy = torch.cat([derive_filter_targets(item, cfg.dt)[..., :2] for item in sessions], dim=0)
    return FilterNormalizerBundle(
        wheel=fit_channel_stats(wheel, cfg.sensor_normalization),
        gps=fit_channel_stats(gps, cfg.sensor_normalization),
        imu=fit_channel_stats(imu, cfg.sensor_normalization),
        target_xy=fit_channel_stats(target_xy, cfg.target_xy_normalization),
    )


def _window_starts(total: int, seq_len: int, stride: int) -> list[int]:
    starts = list(range(0, total - seq_len + 1, stride))
    if starts and starts[-1] != total - seq_len:
        starts.append(total - seq_len)
    return starts


class PureMambaFilterWindowDataset(Dataset):
    def __init__(
        self,
        sessions: list[dict[str, Any]],
        data_cfg: PureMambaFilterDataConfig,
        normalizers: FilterNormalizerBundle,
        seq_len: int,
        stride: int,
        log_debug: bool = True,
    ) -> None:
        self.sessions = sessions
        self.data_cfg = data_cfg
        self.normalizers = normalizers
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.windows: list[dict[str, torch.Tensor | str]] = []
        self._build_windows()
        if log_debug:
            self._log_target_debug()

    def _build_windows(self) -> None:
        for item in self.sessions:
            wheel = self.normalizers.wheel.normalize(extract_wheel_features(item, self.data_cfg.dt))
            gps = self.normalizers.gps.normalize(extract_gps_features(item, self.data_cfg.dt))
            imu = self.normalizers.imu.normalize(extract_imu_features(item, self.data_cfg.dt))
            target_raw = derive_filter_targets(item, self.data_cfg.dt)
            target = self.normalizers.normalize_target(target_raw)
            total = target.shape[0]
            if total < self.seq_len:
                continue
            for start in _window_starts(total, self.seq_len, self.stride):
                end = start + self.seq_len
                init_state = item["initial_state"].float().clone()
                self.windows.append(
                    {
                        "wheel": wheel[start:end],
                        "gps": gps[start:end],
                        "imu": imu[start:end],
                        "target": target[start:end],
                        "initial_state": self.normalizers.normalize_initial_state(init_state),
                        "data_date": str(item.get("data_date", "unknown")),
                    }
                )

    def _log_target_debug(self) -> None:
        if not self.windows:
            return
        target = torch.cat([w["target"][..., :2].reshape(-1, 2) for w in self.windows], dim=0)  # type: ignore[index]
        finite = torch.isfinite(target).all(dim=-1)
        if not finite.any():
            return
        finite_target = target[finite]
        print(
            json.dumps(
                {
                    "event": "pure_mamba_filter_target_debug",
                    "seq_len": self.seq_len,
                    "normalized_target_min": finite_target.min(dim=0).values.tolist(),
                    "normalized_target_max": finite_target.max(dim=0).values.tolist(),
                    "target_center_xy": self.normalizers.target_xy.center.tolist(),
                    "target_scale_xy": self.normalizers.target_xy.scale.tolist(),
                }
            ),
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return self.windows[idx]


def collate_filter_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "wheel": torch.stack([item["wheel"] for item in batch], dim=0),  # type: ignore[list-item]
        "gps": torch.stack([item["gps"] for item in batch], dim=0),  # type: ignore[list-item]
        "imu": torch.stack([item["imu"] for item in batch], dim=0),  # type: ignore[list-item]
        "target": torch.stack([item["target"] for item in batch], dim=0),  # type: ignore[list-item]
        "initial_state": torch.stack([item["initial_state"] for item in batch], dim=0),  # type: ignore[list-item]
        "data_date": [str(item["data_date"]) for item in batch],
    }


def make_loader(dataset: PureMambaFilterWindowDataset, cfg: PureMambaFilterTrainConfig, shuffle: bool) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "shuffle": shuffle,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_filter_batch,
    }
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(dataset, **kwargs)


def prepare_filter_session(
    item: dict[str, Any],
    data_cfg: PureMambaFilterDataConfig,
    normalizers: FilterNormalizerBundle,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    initial_state = normalizers.normalize_initial_state(item["initial_state"].float()).unsqueeze(0).to(device)
    return {
        "wheel": normalizers.wheel.normalize(extract_wheel_features(item, data_cfg.dt)).unsqueeze(0).to(device),
        "gps": normalizers.gps.normalize(extract_gps_features(item, data_cfg.dt)).unsqueeze(0).to(device),
        "imu": normalizers.imu.normalize(extract_imu_features(item, data_cfg.dt)).unsqueeze(0).to(device),
        "initial_state": initial_state,
    }


def _finite_summary(x: torch.Tensor) -> dict[str, float | int | None]:
    finite = torch.isfinite(x)
    values = x[finite]
    if values.numel() == 0:
        return {"min": None, "max": None, "mean": None, "nonfinite_count": int(x.numel())}
    return {
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "nonfinite_count": int(x.numel() - finite.sum().item()),
    }


def sanitize_meter_xy(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x.float(), nan=0.0, posinf=1.0e4, neginf=-1.0e4).clamp(-1.0e4, 1.0e4)


@torch.no_grad()
def predict_filter_session(
    model: torch.nn.Module,
    item: dict[str, Any],
    data_cfg: PureMambaFilterDataConfig,
    normalizers: FilterNormalizerBundle,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    batch = prepare_filter_session(item, data_cfg, normalizers, device)
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        output = model(**batch)
    pred_norm_xy = output["position"][0].float().cpu()
    pred_meter_xy = normalizers.denormalize_xy(pred_norm_xy)
    return sanitize_meter_xy(pred_meter_xy)


@torch.no_grad()
def evaluate_filter_sessions(
    model: torch.nn.Module,
    sessions: list[dict[str, Any]],
    data_cfg: PureMambaFilterDataConfig,
    normalizers: FilterNormalizerBundle,
    device: torch.device,
) -> dict[str, Any]:
    preds = []
    targets = []
    rows = []
    for item in sessions:
        pred_xy = predict_filter_session(model, item, data_cfg, normalizers, device)
        target_xy = sanitize_meter_xy(item["ground_truth"].float())
        diff = pred_xy - target_xy
        rmse = float(torch.sqrt(diff.square().sum(dim=-1).mean()).item())
        pred_stats = _finite_summary(pred_xy)
        print(
            json.dumps(
                {
                    "event": "pure_mamba_filter_eval_debug",
                    "data_date": str(item.get("data_date", "unknown")),
                    "denormalized_prediction_min": pred_stats["min"],
                    "denormalized_prediction_max": pred_stats["max"],
                    "denormalized_prediction_mean": pred_stats["mean"],
                    "prediction_nonfinite_count": pred_stats["nonfinite_count"],
                    "rmse_m": rmse,
                }
            ),
            flush=True,
        )
        preds.append(pred_xy)
        targets.append(target_xy)
        rows.append(
            {
                "data_date": str(item.get("data_date", "unknown")),
                "rmse_m": rmse,
                "trajectory_length": int(target_xy.shape[0]),
            }
        )
    all_preds = torch.cat(preds, dim=0)
    all_targets = torch.cat(targets, dim=0)
    overall = float(torch.sqrt((all_preds - all_targets).square().sum(dim=-1).mean()).item())
    return {
        "overall_rmse_m": overall,
        "per_trajectory": rows,
        "predictions": preds,
        "targets": targets,
    }


@contextmanager
def ema_context(model: torch.nn.Module, ema: EMA | None):
    if ema is None:
        yield
        return
    ema.apply(model)
    try:
        yield
    finally:
        ema.restore(model)


def _stage_epochs(train_cfg: PureMambaFilterTrainConfig) -> list[int]:
    stages = list(train_cfg.curriculum)
    epochs = list(train_cfg.curriculum_epochs)
    if len(epochs) != len(stages):
        epochs = [max(train_cfg.epochs // max(len(stages), 1), 1)] * len(stages)
    if sum(epochs) == train_cfg.epochs:
        return epochs
    total = max(sum(epochs), 1)
    remaining = train_cfg.epochs
    planned = []
    for idx, weight in enumerate(epochs):
        value = max(int(round(train_cfg.epochs * weight / total)), 1)
        if idx == len(epochs) - 1:
            value = remaining
        value = max(min(value, remaining), 0)
        planned.append(value)
        remaining -= value
    return planned


def train_pure_mamba_filter(
    model_cfg: PureMambaFilterConfig,
    data_cfg: PureMambaFilterDataConfig,
    loss_cfg: PureMambaFilterLossConfig,
    optim_cfg: OptimConfig,
    train_cfg: PureMambaFilterTrainConfig,
) -> dict[str, Any]:
    set_seed(train_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_sessions = load_sessions(data_cfg.train_path)
    val_sessions = load_sessions(data_cfg.val_path)
    test_sessions = load_sessions(data_cfg.test_path)
    normalizers = fit_filter_normalizers(train_sessions, data_cfg)

    stage_lengths = list(train_cfg.curriculum)
    stage_epoch_counts = _stage_epochs(train_cfg)
    datasets = [
        PureMambaFilterWindowDataset(
            train_sessions,
            data_cfg,
            normalizers,
            seq_len=seq_len,
            stride=max(int(seq_len * 0.5), 1),
            log_debug=True,
        )
        for seq_len in stage_lengths
    ]
    total_steps = sum(max(math.ceil(len(dataset) / max(train_cfg.batch_size, 1)), 1) * epochs for dataset, epochs in zip(datasets, stage_epoch_counts))

    model = build_pure_mamba_filter(model_cfg).to(device)
    optimizer = build_optimizer(model.parameters(), optim_cfg)
    scheduler = build_scheduler(optimizer, optim_cfg, total_steps=max(total_steps, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.amp and device.type == "cuda")
    ema = EMA(model, train_cfg.ema_decay) if train_cfg.ema else None

    best_rmse = float("inf")
    best_payload = None
    no_improve = 0
    history: list[dict[str, Any]] = []

    for stage_idx, (seq_len, epochs, dataset) in enumerate(zip(stage_lengths, stage_epoch_counts, datasets)):
        if epochs <= 0:
            continue
        loader = make_loader(dataset, train_cfg, shuffle=True)
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            batches = 0
            for batch in loader:
                wheel = batch["wheel"].to(device, non_blocking=True)  # type: ignore[union-attr]
                gps = batch["gps"].to(device, non_blocking=True)  # type: ignore[union-attr]
                imu = batch["imu"].to(device, non_blocking=True)  # type: ignore[union-attr]
                target = batch["target"].to(device, non_blocking=True)  # type: ignore[union-attr]
                initial_state = batch["initial_state"].to(device, non_blocking=True)  # type: ignore[union-attr]
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=train_cfg.amp and device.type == "cuda"):
                    output = model(wheel=wheel, gps=gps, imu=imu, initial_state=initial_state)
                    loss, parts = pure_mamba_filter_loss(output, target, loss_cfg)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = (not train_cfg.amp) or scaler.get_scale() >= old_scale
                if stepped:
                    scheduler.step()
                    if ema is not None:
                        ema.update(model)
                epoch_loss += parts["loss"]
                batches += 1

            with ema_context(model, ema):
                val_metrics = evaluate_filter_sessions(model, val_sessions, data_cfg, normalizers, device)
            val_rmse = val_metrics["overall_rmse_m"]
            row = {
                "stage": stage_idx + 1,
                "seq_len": seq_len,
                "epoch": epoch + 1,
                "train_loss": epoch_loss / max(batches, 1),
                "val_rmse_m": val_rmse,
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            print(json.dumps(row), flush=True)

            checkpoint_payload = {
                "model": model.state_dict(),
                "model_config": asdict(model_cfg),
                "data_config": asdict(data_cfg),
                "loss_config": asdict(loss_cfg),
                "optim_config": asdict(optim_cfg),
                "train_config": asdict(train_cfg),
                "normalizers": normalizers.to_dict(),
                "history": history,
                "val_rmse_m": val_rmse,
            }
            save_checkpoint(output_dir / "last.pt", **checkpoint_payload)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                no_improve = 0
                with ema_context(model, ema):
                    checkpoint_payload["model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_payload = checkpoint_payload
                save_checkpoint(output_dir / "best.pt", **checkpoint_payload)
            else:
                no_improve += 1
                if no_improve >= train_cfg.early_stopping_patience:
                    break
        if no_improve >= train_cfg.early_stopping_patience:
            break

    if best_payload is not None:
        model.load_state_dict(best_payload["model"])
    test_metrics = evaluate_filter_sessions(model, test_sessions, data_cfg, normalizers, device)
    summary = {
        "best_val_rmse_m": best_rmse,
        "test_summary": {
            "overall_rmse_m": test_metrics["overall_rmse_m"],
            "per_trajectory": test_metrics["per_trajectory"],
        },
        "history": history,
        "artifacts": {
            "best_checkpoint": str(output_dir / "best.pt"),
            "last_checkpoint": str(output_dir / "last.pt"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_forward_smoke(model_cfg: PureMambaFilterConfig, device: torch.device) -> None:
    model = build_pure_mamba_filter(model_cfg).to(device)
    batch = {
        "wheel": torch.randn(2, 20, model_cfg.wheel_dim, device=device),
        "gps": torch.randn(2, 20, model_cfg.gps_dim, device=device),
        "imu": torch.randn(2, 20, model_cfg.imu_dim, device=device),
        "initial_state": torch.randn(2, model_cfg.initial_state_dim, device=device),
    }
    batch["gps"][0, 3:6, -1] = 0.0
    out = model(**batch)
    if not torch.isfinite(out["state_estimates"]).all():
        raise RuntimeError("Forward smoke produced non-finite state estimates.")
    print(json.dumps({"event": "forward_smoke_ok", "shape": list(out["state_estimates"].shape)}), flush=True)


def run_single_batch_train_smoke(
    model_cfg: PureMambaFilterConfig,
    loss_cfg: PureMambaFilterLossConfig,
    device: torch.device,
) -> None:
    model = build_pure_mamba_filter(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5)
    wheel = torch.randn(2, 20, model_cfg.wheel_dim, device=device)
    gps = torch.randn(2, 20, model_cfg.gps_dim, device=device)
    imu = torch.randn(2, 20, model_cfg.imu_dim, device=device)
    initial_state = torch.randn(2, model_cfg.initial_state_dim, device=device)
    target = torch.randn(2, 20, model_cfg.output_state_dim, device=device)
    output = model(wheel=wheel, gps=gps, imu=imu, initial_state=initial_state)
    loss, parts = pure_mamba_filter_loss(output, target, loss_cfg)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Single-batch smoke loss is not finite: {parts}")
    loss.backward()
    finite_grads = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not all(bool(item) for item in finite_grads):
        raise RuntimeError("Single-batch smoke produced non-finite gradients.")
    optimizer.step()
    print(json.dumps({"event": "single_batch_train_smoke_ok", "loss": parts["loss"]}), flush=True)


def load_configs(path: str | Path) -> tuple[PureMambaFilterConfig, PureMambaFilterDataConfig, PureMambaFilterLossConfig, OptimConfig, PureMambaFilterTrainConfig]:
    payload = load_yaml(path)
    model_cfg = update_dataclass(PureMambaFilterConfig(), payload.get("model", {}))
    data_cfg = update_dataclass(PureMambaFilterDataConfig(), payload.get("data", {}))
    loss_cfg = update_dataclass(PureMambaFilterLossConfig(), payload.get("loss", {}))
    optim_cfg = update_dataclass(OptimConfig(), payload.get("optim", {}))
    train_cfg = update_dataclass(PureMambaFilterTrainConfig(), payload.get("train", {}))
    train_cfg.curriculum = tuple(train_cfg.curriculum)
    train_cfg.curriculum_epochs = tuple(train_cfg.curriculum_epochs)
    return model_cfg, data_cfg, loss_cfg, optim_cfg, train_cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs" / "pure_mamba_filter.yaml"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    model_cfg, data_cfg, loss_cfg, optim_cfg, train_cfg = load_configs(args.config)
    if args.output_dir:
        train_cfg.output_dir = args.output_dir
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        optim_cfg.lr = args.lr
    if args.amp:
        train_cfg.amp = True
    if args.no_amp:
        train_cfg.amp = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        run_forward_smoke(model_cfg, device)
        run_single_batch_train_smoke(model_cfg, loss_cfg, device)
        return 0

    summary = train_pure_mamba_filter(model_cfg, data_cfg, loss_cfg, optim_cfg, train_cfg)
    print(f"Test RMSE: {summary['test_summary']['overall_rmse_m']:.4f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
