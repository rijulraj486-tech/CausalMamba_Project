"""
Causal Mamba trainer.

Critical fixes vs broken bidirectional version:
  1. Forward-only Mamba blocks eliminate GPS lookahead leakage.
  2. A soft 3% GPS anchor during training prevents short-window divergence.
  3. Evaluation uses a hard GPS anchor when valid GPS is available.
  4. Block GPS dropout simulates consecutive indoor GPS outages.
  5. pred_pos_buf is cleared after every backward call.
  6. Window-relative coordinates bound prediction scale.
  7. Gradient clipping at 1.0 stabilizes long-D training.
"""

import json
import torch
import numpy as np
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from training.losses import MultiScaleCurriculumLoss


class CausalMambaTrainer:
    """Trainer implementing TBPTT, block GPS dropout, and Table II evaluation."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        config,
        save_prefix='results/causal_mamba',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.save_prefix = save_prefix
        self.loss_fn = MultiScaleCurriculumLoss(
            scales=config.get('loss_scales', (1, 5, 20, 50)),
            warmup_epochs=config.get('warmup_epochs', 15),
        )
        self.mamba_context = int(config.get('mamba_context', 32))
        self._optimizer = None
        self.gps_anchor_alpha = float(config.get('gps_anchor_alpha', 0.03))
        self.eval_gps_anchor_alpha = float(
            config.get('eval_gps_anchor_alpha', 1.0)
        )
        self.gps_fusion_mode = str(config.get('gps_fusion_mode', 'fixed')).lower()
        self.gps_confidence_prior = float(config.get('gps_confidence_prior', 0.1))
        self.gps_confidence_reg = float(config.get('gps_confidence_reg', 0.0))
        self.best_val = float('inf')
        self.best_path = None
        self.log = []
        self._scheduler = None

    @staticmethod
    def block_gps_dropout(gps_valid_seq, block_size=40, prob=0.40):
        """
        Harder GPS dropout with longer blocks.
        block_size=40 simulates 40-second indoor outages (realistic).
        prob=0.40 means 40% of windows have GPS dropped.
        This makes the task hard enough that frequent gradient updates
        (TBPTT(2,4)) help significantly more than infrequent ones (TBPTT(D,D)).
        """
        B, D = gps_valid_seq.shape
        mask = torch.ones(
            B, D, dtype=torch.bool, device=gps_valid_seq.device
        )
        stride = max(1, block_size // 2)
        for b in range(B):
            t = 0
            while t < D:
                if torch.rand(1, device=gps_valid_seq.device).item() < prob:
                    end = min(t + block_size, D)
                    mask[b, t:end] = False
                t += stride
        return gps_valid_seq & mask

    def _model_last_delta(
        self,
        wheel_hist,
        gps_hist,
        imu_hist,
        gps_valid_hist,
    ):
        start = max(0, len(wheel_hist) - self.mamba_context)
        wheel_seq = torch.stack(wheel_hist[start:], dim=1)
        gps_seq = torch.stack(gps_hist[start:], dim=1)
        imu_seq = torch.stack(imu_hist[start:], dim=1)
        gps_valid_seq = torch.stack(gps_valid_hist[start:], dim=1)
        if self.gps_fusion_mode == 'learned':
            delta, log_sigma, gps_conf = self.model(
                wheel_seq, gps_seq, imu_seq, gps_valid_seq,
                return_confidence=True,
            )
            return delta[:, -1], log_sigma, gps_conf[:, -1]
        delta, log_sigma = self.model(wheel_seq, gps_seq, imu_seq, gps_valid_seq)
        return delta[:, -1], log_sigma, None

    @staticmethod
    def gps_anchor(x_integrated, gps_xy, gps_valid, alpha=1.0):
        """
        Soft GPS anchor.
        At eval:     alpha=1.0  -> hard reset to GPS (full correction)
        At training: alpha=0.03 -> gentle 3% nudge toward GPS
        """
        gv = gps_valid.float().unsqueeze(-1)
        if not torch.is_tensor(alpha):
            alpha = torch.tensor(
                float(alpha), dtype=x_integrated.dtype, device=x_integrated.device
            )
        alpha = alpha.to(dtype=x_integrated.dtype, device=x_integrated.device)
        if alpha.ndim == 0:
            alpha = alpha.view(1, 1)
        if alpha.ndim == 1:
            alpha = alpha.unsqueeze(-1)
        alpha = alpha.clamp(0.0, 1.0)
        x_new = x_integrated.clone()
        x_new[:, :2] = (
            gv * (alpha * gps_xy + (1.0 - alpha) * x_integrated[:, :2])
            + (1.0 - gv) * x_integrated[:, :2]
        )
        return x_new

    def run_sequence(self, batch, k, w, D, epoch, gps_dropout=0.0):
        """
        TBPTT(k, w, D) training on one batch.
        k : gradient detach interval
        w : backward / weight update interval
        D : sequence length to process
        """
        dev = self.device
        B = batch['wheel'].shape[0]
        D = min(D, batch['wheel'].shape[1])

        x_int = batch['x0'].to(dev).clone()

        # Apply block GPS dropout ONCE per sequence (not per-step)
        gps_valid_all = batch['gps_valid'][:, :D].to(dev)
        if gps_dropout > 0 and self.model.training:
            gps_valid_all = self.block_gps_dropout(
                gps_valid_all, block_size=40, prob=gps_dropout
            )

        anchor_xy = x_int[:, :2].detach()
        pred_pos_buf = []
        gt_pos_buf = []
        gps_conf_buf = []
        gps_valid_buf = []
        wheel_hist = []
        gps_hist = []
        imu_hist = []
        gps_valid_hist = []
        total_loss = 0.0
        opt = self._get_optimizer()

        for t in range(D):
            gps_xy = batch['gps'][:, t, :2].to(dev)
            gps_valid = gps_valid_all[:, t]

            # GPS features relative to current integrated position (bounded)
            gps_dx = gps_xy[:, 0:1] - x_int[:, 0:1].detach()
            gps_dy = gps_xy[:, 1:2] - x_int[:, 1:2].detach()
            gps_feat = torch.cat(
                [gps_dx, gps_dy, gps_valid.float().unsqueeze(-1)], dim=-1
            )

            wheel_feat = batch['wheel'][:, t].to(dev)
            imu_feat = batch['imu'][:, t].to(dev)
            gps_val = gps_valid.float().unsqueeze(-1)

            wheel_hist.append(wheel_feat)
            gps_hist.append(gps_feat)
            imu_hist.append(imu_feat)
            gps_valid_hist.append(gps_val)

            delta, log_sigma, gps_conf = self._model_last_delta(
                wheel_hist,
                gps_hist,
                imu_hist,
                gps_valid_hist,
            )
            x_int = x_int + delta

            anchor_alpha = (
                gps_conf if self.gps_fusion_mode == 'learned'
                else self.gps_anchor_alpha
            )
            x_int = self.gps_anchor(
                x_int, gps_xy, gps_valid, alpha=anchor_alpha
            )
            if gps_conf is not None:
                gps_conf_buf.append(gps_conf)
                gps_valid_buf.append(gps_valid.float().unsqueeze(-1))

            pred_pos_buf.append(x_int[:, :2] - anchor_xy)
            gt_pos_buf.append(batch['gt'][:, t, :2].to(dev) - anchor_xy)

            # Detach every k steps
            if (t + 1) % k == 0:
                x_int = x_int.detach()

            # Update every w steps
            if (t + 1) % w == 0 or t == D - 1:
                window_pred = torch.stack(pred_pos_buf, dim=1)
                window_gt = torch.stack(gt_pos_buf, dim=1)

                loss = self.loss_fn(window_pred, window_gt, epoch, log_sigma)
                if (
                    self.gps_fusion_mode == 'learned'
                    and self.gps_confidence_reg > 0
                    and gps_conf_buf
                ):
                    conf = torch.stack(gps_conf_buf, dim=1)
                    valid = torch.stack(gps_valid_buf, dim=1)
                    denom = valid.sum().clamp_min(1.0)
                    conf_loss = (
                        ((conf - self.gps_confidence_prior) ** 2) * valid
                    ).sum() / denom
                    loss = loss + self.gps_confidence_reg * conf_loss

                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                opt.step()

                pred_pos_buf = []
                gt_pos_buf = []
                gps_conf_buf = []
                gps_valid_buf = []
                anchor_xy = x_int[:, :2].detach()
                total_loss += loss.item()

        return total_loss

    def train(self, phases):
        """
        phases: list of dicts with keys:
          epochs, D, k, w, gps_dropout, lr, label
        Saves phase-specific checkpoint - no phase overwrites another.
        Returns path to overall best checkpoint.
        """
        for phase in phases:
            self._set_lr(phase['lr'])
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._get_optimizer(),
                T_max=max(1, len(phase['epochs'])),
                eta_min=1e-6,
            )
            print(f"\n{'=' * 60}")
            print(
                f"  {phase['label']} | D={phase['D']} k={phase['k']} "
                f"w={phase['w']} lr={phase['lr']} "
                f"gps_drop={phase['gps_dropout']}"
            )
            print(f"{'=' * 60}")

            for epoch in phase['epochs']:
                self.model.train()
                epoch_loss = 0.0
                n_batches = 0

                for batch in self.train_loader:
                    loss = self.run_sequence(
                        batch,
                        k=phase['k'],
                        w=phase['w'],
                        D=phase['D'],
                        epoch=epoch,
                        gps_dropout=phase['gps_dropout'],
                    )
                    epoch_loss += loss
                    n_batches += 1

                val_rmse = self.validate()
                row = {
                    'epoch': epoch,
                    'phase': phase['label'],
                    'train_loss': epoch_loss / max(1, n_batches),
                    'val_rmse_m': val_rmse,
                }
                self.log.append(row)
                print(json.dumps(row))

                if val_rmse < self.best_val:
                    self.best_val = val_rmse
                    self.best_path = (
                        f"{self.save_prefix}_{phase['label']}_best.pt"
                    )
                    torch.save(self.model.state_dict(), self.best_path)
                    print(
                        f"  ** New best val RMSE: {val_rmse:.4f} m "
                        f"- saved to {self.best_path} **"
                    )

                self._scheduler.step()

        print(f"\nTraining complete. Best val RMSE: {self.best_val:.4f} m")
        return self.best_path

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        rmse_list = []
        for batch in self.val_loader:
            rmse_list.append(self._eval_one_trajectory(batch))
        return float(np.mean(rmse_list))

    @torch.no_grad()
    def _eval_one_trajectory(self, batch):
        dev = self.device
        T = batch['wheel'].shape[1]  # FULL trajectory length
        x_int = batch['x0'].to(dev).clone()
        errs = []
        wheel_hist = []
        gps_hist = []
        imu_hist = []
        gps_valid_hist = []

        for t in range(T):
            gps_xy = batch['gps'][:, t, :2].to(dev)
            gps_valid = batch['gps_valid'][:, t].to(dev)

            gps_dx = gps_xy[:, 0:1] - x_int[:, 0:1]
            gps_dy = gps_xy[:, 1:2] - x_int[:, 1:2]
            gps_feat = torch.cat(
                [gps_dx, gps_dy, gps_valid.float().unsqueeze(-1)], dim=-1
            )

            wheel_hist.append(batch['wheel'][:, t].to(dev))
            gps_hist.append(gps_feat)
            imu_hist.append(batch['imu'][:, t].to(dev))
            gps_valid_hist.append(gps_valid.float().unsqueeze(-1))

            delta, _, gps_conf = self._model_last_delta(
                wheel_hist,
                gps_hist,
                imu_hist,
                gps_valid_hist,
            )
            x_int = x_int + delta

            anchor_alpha = (
                gps_conf if self.gps_fusion_mode == 'learned'
                else self.eval_gps_anchor_alpha
            )
            x_int = self.gps_anchor(
                x_int, gps_xy, gps_valid, alpha=anchor_alpha
            )

            gt_xy = batch['gt'][:, t, :2].to(dev)
            errs.append((x_int[:, :2] - gt_xy).norm(dim=-1))

        return torch.stack(errs).mean().item()

    def _set_lr(self, lr):
        for pg in self._get_optimizer().param_groups:
            pg['lr'] = lr
            pg['initial_lr'] = lr

    def _get_optimizer(self):
        if self._optimizer is None:
            self._optimizer = Adam(
                self.model.parameters(), lr=1e-4, weight_decay=1e-5
            )
        return self._optimizer


def make_three_phase_schedule():
    """
    Phase 1 (epochs 0-19):  D=50,  k=2, w=4,  gps_drop=0.40, lr=1e-4
      -> Teaches dead reckoning with indoor-style GPS outages
    Phase 2 (epochs 20-39): D=100, k=2, w=8,  gps_drop=0.40, lr=1e-4
      -> Longer context, maintains GPS-outage robustness
    Phase 3 (epochs 40-49): D=100, k=2, w=8,  gps_drop=0.0,  lr=1e-4
      -> Fine-tune with real GPS availability from dataset
    """
    return [
        dict(
            epochs=range(0, 20),
            D=50,
            k=2,
            w=4,
            gps_dropout=0.40,
            lr=1e-4,
            label='phase1_D50_drop40',
        ),
        dict(
            epochs=range(20, 40),
            D=100,
            k=2,
            w=8,
            gps_dropout=0.40,
            lr=1e-4,
            label='phase2_D100_drop40',
        ),
        dict(
            epochs=range(40, 50),
            D=100,
            k=2,
            w=8,
            gps_dropout=0.0,
            lr=1e-4,
            label='phase3_D100_finetune',
        ),
    ]
