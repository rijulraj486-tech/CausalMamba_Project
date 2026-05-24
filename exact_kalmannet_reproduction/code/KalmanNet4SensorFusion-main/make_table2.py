#!/usr/bin/env python3
"""Build Table II comparison from experiment metric files."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


COLUMNS = [
    "2_4_50",
    "2_8_50",
    "50_50_50",
    "2_4_100",
    "2_8_100",
    "100_100_100",
    "2_4_200",
    "2_8_200",
    "200_200_200",
]

PAPER = {
    "KalmanNet (paper)": {
        "2_4_50": 8.01,
        "2_8_50": 9.00,
        "50_50_50": 11.12,
        "2_4_100": 8.62,
        "2_8_100": 9.20,
        "100_100_100": math.nan,
        "2_4_200": 10.45,
        "2_8_200": 10.45,
        "200_200_200": math.nan,
    }
}


def _read_json_metric(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    for key in ("average_rmse_m", "overall_rmse_m", "test_rmse_m", "rmse", "RMSE"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    nested = payload.get("test_summary")
    if isinstance(nested, dict):
        value = nested.get("overall_rmse_m") or nested.get("average_rmse_m")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _read_log_metric(path: Path) -> float | None:
    if not path.exists():
        return None
    text = path.read_text(errors="ignore")
    patterns = [
        r"test[/_ ]RMSE[^0-9+\-.]*(\d+(?:\.\d+)?)",
        r"RMSE LOSS:\s*(\d+(?:\.\d+)?)",
        r"average_rmse_m['\"]?\s*[:=]\s*(\d+(?:\.\d+)?)",
        r"overall_rmse_m['\"]?\s*[:=]\s*(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return float(matches[-1])
    return None


def read_experiment(prefix: str, tbptt: str) -> float:
    exp_dir = Path("experiments") / f"{prefix}_{tbptt}"
    candidates = [
        exp_dir / "metrics.json",
        exp_dir / "results.json",
        exp_dir / "test_metrics.json",
        exp_dir / "eval_results" / "metrics.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            value = _read_json_metric(candidate)
            if value is not None:
                return value
    value = _read_log_metric(Path("logs") / f"{prefix}_{tbptt}.log")
    return math.nan if value is None else value


def fmt(value: float) -> str:
    return "NaN" if value is None or math.isnan(value) else f"{value:.2f}"


def build_rows() -> dict[str, dict[str, float]]:
    rows = dict(PAPER)
    rows["KalmanNet (ours)"] = {col: read_experiment("knet", col) for col in COLUMNS}
    rows["MambaKalmanNet"] = {col: read_experiment("mamba", col) for col in COLUMNS}
    return rows


def render_table(rows: dict[str, dict[str, float]]) -> str:
    header = [
        "TABLE II - Average RMSE in Meters on Test Dataset",
        "=" * 74,
        "Method              |    D=50           |    D=100          |    D=200",
        "                    | (2,4) (2,8) (D,D) | (2,4) (2,8) (D,D) | (2,4) (2,8) (D,D)",
        "-" * 74,
    ]
    body = []
    for name, values in rows.items():
        body.append(
            f"{name:<19} | "
            f"{fmt(values['2_4_50']):>5} {fmt(values['2_8_50']):>5} {fmt(values['50_50_50']):>6} | "
            f"{fmt(values['2_4_100']):>5} {fmt(values['2_8_100']):>5} {fmt(values['100_100_100']):>6} | "
            f"{fmt(values['2_4_200']):>5} {fmt(values['2_8_200']):>5} {fmt(values['200_200_200']):>6}"
        )
    return "\n".join([*header, *body, "=" * 74, "Bold = best result per column"])


def main() -> int:
    rows = build_rows()
    table = render_table(rows)
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "table2.txt").write_text(table + "\n", encoding="utf-8")
    with (output_dir / "table2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Method", *COLUMNS])
        for name, values in rows.items():
            writer.writerow([name, *[fmt(values[col]) for col in COLUMNS]])
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
