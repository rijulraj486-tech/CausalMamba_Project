"""Optimizers and schedulers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


class Lion(torch.optim.Optimizer):
    """Minimal Lion optimizer implementation."""

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99), weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if wd:
                    p.data.mul_(1 - lr * wd)
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


@dataclass
class OptimConfig:
    optimizer: str = "adamw"
    lr: float = 2e-4
    weight_decay: float = 1e-4
    scheduler: str = "warmup_cosine"
    warmup_frac: float = 0.08
    min_lr_scale: float = 0.05


def build_optimizer(params, cfg: OptimConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "lion":
        return Lion(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: OptimConfig, total_steps: int):
    total_steps = max(int(total_steps), 1)
    if cfg.scheduler == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    if cfg.scheduler == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    if cfg.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=total_steps)
    if cfg.scheduler in {"cosine", "warmup_cosine"}:
        warmup = max(int(total_steps * cfg.warmup_frac), 1) if cfg.scheduler == "warmup_cosine" else 0

        def schedule(step: int) -> float:
            if step < warmup:
                return float(step + 1) / float(warmup)
            progress = (step - warmup) / max(total_steps - warmup, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cfg.min_lr_scale + (1.0 - cfg.min_lr_scale) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")
