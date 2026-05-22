import os
import json
import torch
from torch.utils.data import DataLoader

from datasets.nclt_fusion_dataset import NCLTFusionDataset
from models.navigation_models import CausalMambaNavigator
from training.trainer import CausalMambaTrainer
from evaluation.evaluator import evaluate_table2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SETTINGS = [
    (50,  "2,4"),
    (50,  "2,8"),
    (50,  "D,D"),
    (100, "2,4"),
    (100, "2,8"),
    (100, "D,D"),
    (200, "2,4"),
    (200, "2,8"),
    (200, "D,D"),
]

test_ds = NCLTFusionDataset(
    data_path="data/nclt_1hz.pt",
    split="test",
    D=None,
)

test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
)

results = {}

for D, tbptt in SETTINGS:
    ckpt = f"results/table2_runs/D{D}_{tbptt.replace(',', '_')}/best.pt"

    if not os.path.exists(ckpt):
        print(f"Missing checkpoint: {ckpt}")
        continue

    model = CausalMambaNavigator(
        d_model=96,
        d_state=16,
        n_layers=2,
    ).to(DEVICE)

    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)

    out = evaluate_table2(
        model=model,
        test_loader=test_loader,
        device=DEVICE,
        trainer_cls=CausalMambaTrainer,
        model_name=f"Causal-Mamba D={D} TBPTT={tbptt}",
        paper_ref="KalmanNet",
        out_json=None,
        mamba_context=32,
        eval_gps_anchor_alpha=0.03,
        gps_fusion_mode="fixed",
    )

    results[f"D{D}_{tbptt}"] = out["avg_rmse"]

with open("results/full_table2.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved results/full_table2.json")
