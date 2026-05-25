import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slimmamba_experiment.configs.config import Config
from slimmamba_experiment.datasets.dataset import NCLTDataset
from slimmamba_experiment.models.model import SlimMambaNav


def position_errors(pred, target):
    return torch.linalg.norm(pred[:, :, :2] - target[:, :, :2], dim=-1)


@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()
    all_errors = []
    final_errors = []
    abs_state_errors = []

    for batch in loader:
        wheel, gps, imu, target, x0 = [b.to(device) for b in batch]
        pred = model.predict_trajectory(wheel, gps, imu, x0)
        err = position_errors(pred, target)
        all_errors.append(err.reshape(-1).cpu())
        final_errors.append(err[:, -1].cpu())
        abs_state_errors.append((pred - target).abs().reshape(-1, 6).cpu())

    errors = torch.cat(all_errors)
    state_abs = torch.cat(abs_state_errors)
    return {
        "rmse": float(torch.sqrt(torch.mean(errors.square())).item()),
        "mae": float(errors.mean().item()),
        "ate": float(torch.cat(final_errors).mean().item()),
        "state_mae": [float(v) for v in state_abs.mean(dim=0)],
    }


@torch.no_grad()
def save_trajectory_plot(model, loader, device, output_path):
    import matplotlib.pyplot as plt

    model.eval()
    wheel, gps, imu, target, x0 = next(iter(loader))
    wheel, gps, imu, target, x0 = [b.to(device) for b in (wheel, gps, imu, target, x0)]
    pred = model.predict_trajectory(wheel, gps, imu, x0)

    pred_xy = pred[0, :, :2].cpu()
    target_xy = target[0, :, :2].cpu()
    gps_xy = gps[0, :, :2].cpu()
    gps_valid = gps[0, :, 2].cpu() > 0.5

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 8))
    plt.plot(target_xy[:, 0], target_xy[:, 1], label="Ground truth", linewidth=2)
    plt.plot(pred_xy[:, 0], pred_xy[:, 1], label="SlimMamba-Nav", linewidth=2)
    if gps_valid.any():
        plt.scatter(
            gps_xy[gps_valid, 0],
            gps_xy[gps_valid, 1],
            label="Valid GPS",
            s=8,
            alpha=0.45,
        )
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def load_model(cfg, device):
    model = SlimMambaNav.from_config(cfg).to(device)
    state = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    cfg.ensure_dirs()
    if args.checkpoint:
        cfg.checkpoint_path = args.checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)
    loader = NCLTDataset(cfg, split=args.split).get_loader(shuffle=False)
    metrics = evaluate_metrics(model, loader, device)

    out_path = Path(cfg.results_dir) / f"{args.split}_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    if args.plot:
        save_trajectory_plot(
            model,
            loader,
            device,
            Path(cfg.results_dir) / f"{args.split}_trajectory.png",
        )


if __name__ == "__main__":
    main()
