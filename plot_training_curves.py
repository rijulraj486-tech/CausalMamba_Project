import re
from pathlib import Path
import matplotlib.pyplot as plt

log_path = Path("evaluation_results/D50_2_4_plot/train.log")

text = log_path.read_text()

pattern = r'\{"epoch":\s*(\d+).*?"train_loss":\s*([0-9.]+),\s*"val_rmse_m":\s*([0-9.]+)\}'

matches = re.findall(pattern, text)

if not matches:
    raise RuntimeError("No epoch data found in train.log")

epochs = [int(m[0]) for m in matches]
train_loss = [float(m[1]) for m in matches]
val_rmse = [float(m[2]) for m in matches]

best_idx = val_rmse.index(min(val_rmse))
best_epoch = epochs[best_idx]
best_val = val_rmse[best_idx]

print(f"Best validation RMSE: {best_val:.4f} m at epoch {best_epoch}")

# Training loss plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Causal-Mamba Training Loss (D=50, k=2, w=4)")
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_results/D50_2_4_plot/training_loss.png", dpi=300)
plt.close()

# Validation RMSE plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, val_rmse, linewidth=2)
plt.scatter([best_epoch], [best_val], s=80)
plt.xlabel("Epoch")
plt.ylabel("Validation RMSE (m)")
plt.title("Causal-Mamba Validation RMSE (D=50, k=2, w=4)")
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_results/D50_2_4_plot/validation_rmse.png", dpi=300)
plt.close()

print("Plots saved successfully.")
