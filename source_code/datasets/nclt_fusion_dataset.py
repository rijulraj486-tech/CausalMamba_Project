"""
NCLT dataset loader for causal Mamba navigation.
Val/test splits return FULL trajectory length (not D steps).
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset


DEFAULT_SPLIT_DIR = Path(
    "radar_longseq_benchmark/Radar_Tracking_Benchmark/"
    "data/nclt_fusion_michigan_rtk_full"
)


def _split_file(root, split):
    root = Path(root)
    candidates = [root / f"{split}.pt", root / f"{split}_rtk.pt"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find {split}.pt or {split}_rtk.pt under {root}"
    )


def load_trajectories(path, split):
    path = Path(path)
    if path.exists() and path.is_dir():
        return torch.load(
            _split_file(path, split), map_location="cpu", weights_only=False
        )
    if path.exists():
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and split in raw:
            return raw[split]
        if isinstance(raw, list):
            return raw
        raise KeyError(f"Split {split!r} not found in {path}")
    if DEFAULT_SPLIT_DIR.exists():
        print(
            f"Data path {path} not found; using {DEFAULT_SPLIT_DIR} "
            f"for split {split}."
        )
        return torch.load(
            _split_file(DEFAULT_SPLIT_DIR, split),
            map_location="cpu",
            weights_only=False,
        )
    raise FileNotFoundError(path)


def _fill_missing_xy(x):
    valid = torch.isfinite(x).all(dim=-1)
    filled = x.clone().float()
    if filled.numel() == 0:
        return filled, valid
    valid_idx = torch.where(valid)[0]
    if len(valid_idx) == 0:
        return torch.zeros_like(filled).float(), valid
    if not valid[0]:
        filled[0] = filled[valid_idx[0]]
    for t in range(1, filled.shape[0]):
        if not valid[t]:
            filled[t] = filled[t - 1]
    return torch.nan_to_num(filled, nan=0.0), valid


def _causal_gps_filter(gps, valid, mode='none', window=5, motion_delta=None):
    """Filter GPS causally using only current/past samples."""
    mode = str(mode or 'none').lower()
    if mode == 'none':
        return gps

    window = max(1, int(window))
    filtered = gps.clone().float()
    history = []

    if mode in {'moving_avg', 'median'}:
        for t in range(gps.shape[0]):
            if bool(valid[t]):
                history.append(gps[t].float())
                if len(history) > window:
                    history = history[-window:]
                hist = torch.stack(history, dim=0)
                if mode == 'moving_avg':
                    filtered[t] = hist.mean(dim=0)
                else:
                    filtered[t] = hist.median(dim=0).values
            elif t > 0:
                filtered[t] = filtered[t - 1]
        return filtered

    if mode == 'kalman':
        # Lightweight causal motion-compensated Kalman filter for 1 Hz GPS.
        # The prediction step uses wheel/heading odometry when supplied, so
        # filtering GPS does not lag behind a moving vehicle like raw XY
        # smoothing does.
        q = 4.0
        r = 25.0
        x = gps[0].float().clone()
        p = torch.ones(2, dtype=torch.float32, device=gps.device) * r
        for t in range(gps.shape[0]):
            if t > 0 and motion_delta is not None:
                x = x + motion_delta[t].to(dtype=x.dtype, device=x.device)
            p = p + q
            if bool(valid[t]):
                z = gps[t].float()
                k = p / (p + r)
                x = x + k * (z - x)
                p = (1.0 - k) * p
            filtered[t] = x
        return filtered

    raise ValueError(
        f"Unknown gps_filter={mode!r}; expected none, moving_avg, median, kalman"
    )


def _wrap_to_pi(angle):
    return (angle + torch.pi) % (2.0 * torch.pi) - torch.pi


def _derive_gt_from_session(traj):
    pos = traj['ground_truth'].float()
    theta = traj.get('theta_gt')
    if theta is None:
        imu = traj.get('imu', traj.get('imu_sensor'))
        if imu is not None and imu.shape[-1] > 2:
            theta = imu[:, 2]
        else:
            theta = pos.new_zeros(pos.shape[0])
    theta = theta.float().reshape(-1)

    initial = traj.get('initial_state', torch.zeros(6)).float()[:6]
    vel = torch.zeros_like(pos)
    omega = torch.zeros_like(theta)
    if pos.shape[0] > 1:
        vel[1:] = pos[1:] - pos[:-1]
        vel[0] = initial[2:4]
        omega[1:] = _wrap_to_pi(theta[1:] - theta[:-1])
        omega[0] = float(initial[5])
    return torch.cat([pos, vel, theta[:, None], omega[:, None]], dim=-1)


class NCLTFusionDataset(Dataset):
    """
    Returns one trajectory per __getitem__.

    Each item is a dict:
      x0         : (6,)    initial state
      wheel      : (T, 7)  [vl,vr,vc,vdiff,sin_theta,cos_theta,omega]
      gps        : (T, 2)  [gps_x, gps_y]
      gps_valid  : (T,)    bool
      imu        : (T, 4)  [theta,omega,sin_theta,cos_theta]
      gt         : (T, 6)  ground truth state
      traj_id    : str

    Train: optionally chunked to length D.
    Val/test: always full trajectory length.
    """

    def __init__(
        self,
        data_path,
        split='train',
        D=None,
        samples_per_traj=1,
        gps_filter='none',
        gps_filter_window=5,
    ):
        self.trajs = load_trajectories(data_path, split)
        self.split = split
        self.D = D
        self.samples_per_traj = max(1, int(samples_per_traj))
        self.gps_filter = str(gps_filter or 'none').lower()
        self.gps_filter_window = max(1, int(gps_filter_window))

    def __len__(self):
        if self.split == 'train' and self.D is not None:
            return len(self.trajs) * self.samples_per_traj
        return len(self.trajs)

    def __getitem__(self, idx):
        traj_idx = idx % len(self.trajs)
        traj = self.trajs[traj_idx]

        if 'wheel_left' in traj:
            vl = traj['wheel_left'].float()
            vr = traj['wheel_right'].float()
            theta = traj['imu_theta'].float()
            omega = traj['imu_omega'].float()
            gps = torch.stack([traj['gps_x'], traj['gps_y']], dim=-1).float()
            gps_valid = traj['gps_valid'].bool()
            gt = traj['gt'].float()
            x0 = gt[0].clone()
        else:
            if 'filtered_wheel' in traj:
                wheel_src = traj['filtered_wheel'].float()
            else:
                wheel_src = traj['wheel'][..., :2].float()
            vl = wheel_src[:, 0]
            vr = wheel_src[:, 1]
            imu_src = traj.get('imu', traj.get('imu_sensor')).float()
            theta = (
                imu_src[:, 2]
                if imu_src.shape[-1] > 2
                else vl.new_zeros(vl.shape[0])
            )
            omega = (
                imu_src[:, 3]
                if imu_src.shape[-1] > 3
                else vl.new_zeros(vl.shape[0])
            )
            gps_src = (
                traj['filtered_gps'].float()
                if 'filtered_gps' in traj
                else traj['gps'].float()
            )
            gps, gps_valid = _fill_missing_xy(gps_src)
            gt = _derive_gt_from_session(traj)
            x0 = traj.get('initial_state', gt[0]).float()[:6].clone()
            x0[:2] = gt[0, :2]

        vc = 0.5 * (vl + vr)
        vdiff = vr - vl
        motion_delta = torch.stack([vc * torch.cos(theta), vc * torch.sin(theta)], dim=-1)

        gps = _causal_gps_filter(
            gps,
            gps_valid,
            mode=self.gps_filter,
            window=self.gps_filter_window,
            motion_delta=motion_delta,
        )

        wheel = torch.stack(
            [vl, vr, vc, vdiff, torch.sin(theta), torch.cos(theta), omega],
            dim=-1,
        ).float()
        imu = torch.stack(
            [theta, omega, torch.sin(theta), torch.cos(theta)], dim=-1
        ).float()

        item = dict(
            x0=x0.float(),
            wheel=wheel,
            gps=gps.float(),
            gps_valid=gps_valid.bool(),
            imu=imu,
            gt=gt.float(),
            traj_id=traj.get('traj_id', traj.get('data_date', str(traj_idx))),
        )

        # Train: chunk to D; val/test: full length
        if self.split == 'train' and self.D is not None:
            T = wheel.shape[0]
            if T > self.D:
                start = torch.randint(0, T - self.D, (1,)).item()
                for k in ['wheel', 'gps', 'gps_valid', 'imu', 'gt']:
                    item[k] = item[k][start:start + self.D]
                item['x0'] = item['gt'][0].clone()

        return item
