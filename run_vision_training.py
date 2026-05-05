import random

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

from dataset.openface_dataset import make_loaders
from model.frame_train import run

ROOT       = "dataset/UR_LYING_Deception_Dataset/openface_raw"
BATCH_SIZE = 16
LR         = 1e-3
HIDDEN     = 64
NUM_LAYERS = 1
POOLING    = "attention"
PATIENCE   = 10
DEVICE     = "mps"

BIGRU_EPOCHS       = 40
BIGRU_DROPOUT      = 0.5
BIGRU_WEIGHT_DECAY = 1e-3

BILSTM_EPOCHS       = 30
BILSTM_DROPOUT      = 0.4
BILSTM_WEIGHT_DECAY = 1e-4

train_loader, val_loader, test_loader, d_in = make_loaders(
    root_dir=ROOT,
    val_frac=0.2,
    batch_size=BATCH_SIZE,
)

print(
    f"Dataset  |  Train batches: {len(train_loader)}"
    f"  Val batches: {len(val_loader)}"
    f"  Test batches: {len(test_loader)}"
    f"  d_in: {d_in}\n"
)

# Train BiGRU 
bigru_result = run(
    train_loader=train_loader,
    val_loader=val_loader,
    d_in=d_in,
    model_type="bigru",
    device=DEVICE,
    epochs=BIGRU_EPOCHS,
    lr=LR,
    weight_decay=BIGRU_WEIGHT_DECAY,
    hidden=HIDDEN,
    num_layers=NUM_LAYERS,
    dropout=BIGRU_DROPOUT,
    pooling=POOLING,
    save_path="best_vision_bigru.pt",
    patience=PATIENCE,
    plot_prefix="vision_bigru",
)

# Train BiLSTM 
bilstm_result = run(
    train_loader=train_loader,
    val_loader=val_loader,
    d_in=d_in,
    model_type="bilstm",
    device=DEVICE,
    epochs=BILSTM_EPOCHS,
    lr=LR,
    weight_decay=BILSTM_WEIGHT_DECAY,
    hidden=HIDDEN,
    num_layers=NUM_LAYERS,
    dropout=BILSTM_DROPOUT,
    pooling=POOLING,
    save_path="best_vision_bilstm.pt",
    patience=PATIENCE,
    plot_prefix="vision_bilstm",
)

bigru_acc  = bigru_result["best_val_acc"]
bilstm_acc = bilstm_result["best_val_acc"]
winner     = "BiGRU" if bigru_acc >= bilstm_acc else "BiLSTM"

print("=" * 45)
print(" Vision model comparison")
print("=" * 45)
print(f"  BiGRU  best val acc : {bigru_acc:.4f}")
print(f"  BiLSTM best val acc : {bilstm_acc:.4f}")
print(f"  Winner              : {winner}")
print("=" * 45)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(bigru_result["history"]["val_loss"],  label="BiGRU")
ax1.plot(bilstm_result["history"]["val_loss"], label="BiLSTM")
ax1.set_title("Validation Loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.legend(); ax1.grid()

ax2.plot(bigru_result["history"]["val_acc"],  label="BiGRU")
ax2.plot(bilstm_result["history"]["val_acc"], label="BiLSTM")
ax2.set_title("Validation Accuracy")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.legend(); ax2.grid()

fig.suptitle("Vision: BiGRU vs BiLSTM", fontsize=13)
fig.tight_layout()
fig.savefig("vision.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("\nPlots saved:")
print("  vision_bigru_curves.png")
print("  vision_bilstm_curves.png")
print("  vision_comparison.png")
print("\nTraining complete!")
