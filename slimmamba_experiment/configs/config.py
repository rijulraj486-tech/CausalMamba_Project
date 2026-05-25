from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    # Model
    d_embed: int = 64
    d_mamba: int = 128
    n_mamba: int = 4
    d_state_ssm: int = 32
    d_conv: int = 4
    expand: int = 2
    use_mamba: bool = True

    # TBPTT(k, w, D)
    tbptt_k: int = 2
    tbptt_w: int = 4
    seq_len: int = 50

    # Training
    epochs: int = 50
    batch_size: int = 256
    eval_batch_size: int = 1
    samples_per_traj: int = 64
    num_workers: int = 0
    lr: float = 1e-3
    lr_min: float = 1e-5
    wd: float = 1e-4
    grad_clip: float = 1.0
    pos_weight: float = 2.0
    seed: int = 7

    # Dataset
    data_dir: str = str(PROJECT_ROOT / "data" / "nclt_1hz.pt")
    gps_filter: str = "none"
    gps_filter_window: int = 5

    # Experiment outputs stay isolated from the stable project.
    experiment_dir: str = str(EXPERIMENT_ROOT)
    checkpoint_path: str = str(EXPERIMENT_ROOT / "checkpoints" / "slim_mamba_best.pt")
    log_dir: str = str(EXPERIMENT_ROOT / "logs")
    results_dir: str = str(EXPERIMENT_ROOT / "results")

    def ensure_dirs(self) -> None:
        Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
