import torch
from torch.utils.data import DataLoader, Dataset

try:
    from .nclt_fusion_dataset import NCLTFusionDataset
except ImportError:
    from nclt_fusion_dataset import NCLTFusionDataset


class NCLTDataset(Dataset):
    """
    Tuple adapter around the isolated NCLT fusion loader.

    Returns:
      wheel:  (T, 7)
      gps:    (T, 3) with [gps_x, gps_y, gps_valid]
      imu:    (T, 4)
      target: (T, 6)
      x0:     (6,)
    """

    def __init__(self, cfg, split="train"):
        d_steps = cfg.seq_len if split == "train" else None
        self.split = split
        self.cfg = cfg
        self.base = NCLTFusionDataset(
            data_path=cfg.data_dir,
            split=split,
            D=d_steps,
            samples_per_traj=cfg.samples_per_traj,
            gps_filter=cfg.gps_filter,
            gps_filter_window=cfg.gps_filter_window,
        )

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        gps_valid = item["gps_valid"].float().unsqueeze(-1)
        gps = item["gps"].float()
        if gps.shape[-1] == 2:
            gps = torch.cat([gps, gps_valid], dim=-1)
        return (
            item["wheel"].float(),
            gps.float(),
            item["imu"].float(),
            item["gt"].float(),
            item["x0"].float(),
        )

    def get_loader(self, shuffle=None):
        if shuffle is None:
            shuffle = self.split == "train"
        batch_size = self.cfg.batch_size if self.split == "train" else self.cfg.eval_batch_size
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=self.split == "train",
        )
