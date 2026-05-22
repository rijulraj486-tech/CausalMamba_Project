"""Multi-scale curriculum loss for causal Mamba navigation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCurriculumLoss(nn.Module):
    """
    Supervises position accuracy at 1, 5, 20, 50 step horizons.
    Short scales force accurate local dead reckoning.
    Long scales force trajectory-level accuracy.
    Curriculum: long-scale losses phase in gradually over warmup_epochs.
    """

    def __init__(self, scales=(1, 5, 20, 50), warmup_epochs=15):
        super().__init__()
        self.scales = scales
        self.warmup_epochs = warmup_epochs

    def forward(self, pred_pos, gt_pos, epoch, log_sigma=None):
        """
        pred_pos  : (B, T, 2)
        gt_pos    : (B, T, 2)
        epoch     : int
        log_sigma : (6,) optional
        """
        total = torch.zeros(1, device=pred_pos.device, requires_grad=True)
        total = total + 0.0
        T = pred_pos.shape[1]

        for scale in self.scales:
            if T <= scale:
                p = pred_pos[:, -1:, :]
                g = gt_pos[:, -1:, :]
            else:
                idx = torch.arange(0, T, scale, device=pred_pos.device)
                if idx[-1] != T - 1:
                    idx = torch.cat(
                        [idx, torch.tensor([T - 1], device=pred_pos.device)]
                    )
                p = pred_pos.index_select(1, idx)
                g = gt_pos.index_select(1, idx)
            scale_loss = F.huber_loss(p, g, delta=5.0)
            weight = min(
                1.0,
                (epoch + 1) / max(1, self.warmup_epochs * scale / 5),
            )
            total = total + weight * scale_loss

        # Velocity consistency: penalise impossible position jumps
        pred_disp = pred_pos[:, 1:] - pred_pos[:, :-1]
        gt_disp = gt_pos[:, 1:] - gt_pos[:, :-1]
        if pred_disp.numel() > 0:
            total = total + 0.5 * F.huber_loss(pred_disp, gt_disp, delta=2.0)
        total = total + 0.5 * F.huber_loss(
            pred_pos[:, -1, :], gt_pos[:, -1, :], delta=5.0
        )

        if log_sigma is not None:
            total = total + 0.01 * log_sigma[:2].abs().sum()

        return total
