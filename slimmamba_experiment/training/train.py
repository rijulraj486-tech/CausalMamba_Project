import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from slimmamba_experiment.configs.config import Config
from slimmamba_experiment.datasets.dataset import NCLTDataset
from slimmamba_experiment.evaluation.evaluate import evaluate_metrics
from slimmamba_experiment.models.model import SlimMambaNav
from slimmamba_experiment.training.loss import MultiScaleLoss


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tbptt_train_step(model, optimizer, loss_fn, wheel, gps, imu, target, x0, epoch, cfg):
    """
    TBPTT(k=2, w=4, D=50): update every w steps and detach the propagated
    state every k windows.
    """

    _, steps, _ = wheel.shape
    k = cfg.tbptt_k
    w = cfg.tbptt_w
    total_loss = 0.0
    n_updates = 0
    x_current = x0.detach()

    for t_start in range(0, steps, w):
        t_end = min(t_start + w, steps)
        window_idx = t_start // w
        if window_idx > 0 and window_idx % k == 0:
            x_current = x_current.detach()

        pred_states, _, delta_logvar = model(
            wheel[:, t_start:t_end],
            gps[:, t_start:t_end],
            imu[:, t_start:t_end],
            x_current,
        )
        loss = loss_fn(
            pred_states,
            target[:, t_start:t_end],
            delta_logvar,
            epoch,
            t_start=t_start,
            total_steps=steps,
        )

        if loss is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        x_current = pred_states[:, -1].detach()
        if loss is not None:
            total_loss += float(loss.item())
            n_updates += 1

    return total_loss / max(n_updates, 1)


def train(cfg):
    cfg.ensure_dirs()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SlimMambaNav.from_config(cfg).to(device)
    loss_fn = MultiScaleLoss(pos_weight=cfg.pos_weight)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr_min)

    train_loader = NCLTDataset(cfg, split="train").get_loader()
    val_loader = NCLTDataset(cfg, split="val").get_loader(shuffle=False)
    best_val_rmse = float("inf")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            wheel, gps, imu, target, x0 = [b.to(device) for b in batch]
            loss = tbptt_train_step(
                model,
                optimizer,
                loss_fn,
                wheel,
                gps,
                imu,
                target,
                x0,
                epoch,
                cfg,
            )
            epoch_loss += loss

        scheduler.step()
        metrics = evaluate_metrics(model, val_loader, device)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        lr = scheduler.get_last_lr()[0]
        row = {"epoch": epoch, "loss": avg_loss, "lr": lr, **metrics}
        history.append(row)
        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} "
            f"| val_rmse={metrics['rmse']:.3f}m | lr={lr:.2e}"
        )

        if metrics["rmse"] < best_val_rmse:
            best_val_rmse = metrics["rmse"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.__dict__,
                    "epoch": epoch,
                    "val_metrics": metrics,
                },
                cfg.checkpoint_path,
            )
            print(f"  saved checkpoint (val_rmse={best_val_rmse:.3f}m)")

        history_path = Path(cfg.log_dir) / "train_history.json"
        history_path.write_text(json.dumps(history, indent=2) + "\n")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-mamba", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.no_mamba:
        cfg.use_mamba = False
    train(cfg)


if __name__ == "__main__":
    main()
