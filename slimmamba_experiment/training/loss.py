import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroscedasticHuberLoss(nn.Module):
    """
    Proper heteroscedastic Gaussian NLL — bounded below, no mode collapse.

    L = 0.5 * [ residual^2 * exp(-logvar) + logvar ]

    Bounded because:
      logvar -> -inf : exp(-logvar) -> +inf, first term dominates
      logvar -> +inf : second term dominates
    Minimum at logvar = log(residual^2).

    The original pseudo-Huber + 0.5*logvar was UNBOUNDED below because
    the Huber is capped at 1 but logvar -> -inf has no floor.
    """
    def __init__(self, pos_weight=2.0, delta_scale=1.0):
        super().__init__()
        self.pos_weight  = pos_weight
        self.delta_scale = delta_scale

    def forward(self, pred_states, target_states, delta_logvar):
        residual = pred_states - target_states          # (B, T, 6)

        # Hard clamp: sigma in [exp(-3), exp(3)] = [0.05, 20.0]
        logvar = delta_logvar.clamp(-6.0, 6.0)

        # Gaussian NLL — cannot go below minimum
        nll = 0.5 * (residual ** 2 * torch.exp(-logvar) + logvar)

        # Up-weight position dims px, py
        weights = torch.ones(6, device=pred_states.device)
        weights[0] = self.pos_weight
        weights[1] = self.pos_weight

        return (nll * weights).mean()


class MultiScaleLoss(nn.Module):
    """
    Curriculum loss — phases determined by epoch:
      epoch  1-10 : T/4  (learn local motion)
      epoch 11-25 : T/2  (medium range)
      epoch 26+   : T    (full sequence + GPS outage correction)
    """
    def __init__(self, pos_weight=2.0):
        super().__init__()
        self.base_loss = HeteroscedasticHuberLoss(pos_weight=pos_weight)

    def forward(self, pred, target, logvar, epoch, t_start=0, total_steps=None):
        T = pred.shape[1]
        scale_T = int(total_steps or T)
        if epoch <= 10:
            active_end = max(scale_T // 4, 1)
        elif epoch <= 25:
            active_end = max(scale_T // 2, 1)
        else:
            active_end = scale_T

        local_end = max(min(active_end - int(t_start), T), 0)
        if local_end == 0:
            return None

        return self.base_loss(
            pred[:, :local_end],
            target[:, :local_end],
            logvar[:, :local_end]
        )
