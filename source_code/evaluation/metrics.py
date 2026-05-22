"""Benchmark-compatible trajectory metrics."""

from __future__ import annotations

import torch


PREDICTION_CLAMP_MIN = -1.0e4
PREDICTION_CLAMP_MAX = 1.0e4


def finite_summary(tensor: torch.Tensor) -> dict[str, float | int | None]:
    tensor = tensor.detach().float()
    finite = torch.isfinite(tensor)
    finite_values = tensor[finite]
    nonfinite_count = int(tensor.numel() - finite.sum().item())
    if finite_values.numel() == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "nonfinite_count": nonfinite_count,
            "numel": int(tensor.numel()),
        }
    return {
        "min": float(finite_values.min().item()),
        "max": float(finite_values.max().item()),
        "mean": float(finite_values.mean().item()),
        "nonfinite_count": nonfinite_count,
        "numel": int(tensor.numel()),
    }


def sanitize_predictions(pred_xy: torch.Tensor) -> torch.Tensor:
    pred_xy = pred_xy.float()
    pred_xy = torch.nan_to_num(
        pred_xy,
        nan=0.0,
        posinf=PREDICTION_CLAMP_MAX,
        neginf=PREDICTION_CLAMP_MIN,
    )
    return pred_xy.clamp(PREDICTION_CLAMP_MIN, PREDICTION_CLAMP_MAX)


def sanitize_targets(target_xy: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(
        target_xy.float(),
        nan=0.0,
        posinf=PREDICTION_CLAMP_MAX,
        neginf=PREDICTION_CLAMP_MIN,
    )


def sequence_rmse_xy(pred_xy: torch.Tensor, target_xy: torch.Tensor) -> float:
    pred_xy = sanitize_predictions(pred_xy).double()
    target_xy = sanitize_targets(target_xy).double()
    return float(torch.sqrt(((pred_xy - target_xy) ** 2).sum(dim=-1).mean()).item())


def gps_only_rmse(session: dict) -> float:
    return sequence_rmse_xy(session["gps"].float(), session["ground_truth"].float())


def table_markdown(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)
