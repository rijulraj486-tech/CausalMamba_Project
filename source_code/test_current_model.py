import torch
from torch.utils.data import DataLoader

from models.navigation_models import CausalMambaNavigator
from training.trainer import CausalMambaTrainer
from datasets.nclt_fusion_dataset import NCLTFusionDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "loss_scales": (1, 5, 20, 50),
    "warmup_epochs": 15,
    "mamba_context": 32,
    "gps_anchor_alpha": 0.15,
    "eval_gps_anchor_alpha": 1.0,
    "gps_fusion_mode": "fixed",
    "gps_confidence_prior": 0.5,
    "gps_confidence_reg": 0.0,
}

model = CausalMambaNavigator(
    d_model=96,
    n_layers=2,
).to(device)

ckpt = torch.load(
    "results/stable_clean/causal_causal_mamba_d96_single_D50_k2_w4_best.pt",
    map_location=device,
)

model.load_state_dict(ckpt)

dummy_train = NCLTFusionDataset(
    data_path="data/nclt_1hz.pt",
    split="train",
    D=50,
    samples_per_traj=1,
)

test_ds = NCLTFusionDataset(
    data_path="data/nclt_1hz.pt",
    split="test",
    D=None,
    samples_per_traj=1,
)

dummy_train_loader = DataLoader(dummy_train, batch_size=1, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

trainer = CausalMambaTrainer(
    model=model,
    train_loader=dummy_train_loader,
    val_loader=test_loader,
    device=device,
    config=config,
    save_prefix="results/test_eval",
)

rmse = trainer.validate()

print("\n" + "=" * 60)
print(f"STRICT TEST RMSE: {rmse:.4f} m")
print("=" * 60)
