"""
Entry point for training the BiLSTM deception classifier.

Usage:
    uv run python run_training.py
"""

import matplotlib

matplotlib.use("Agg")

from typing import Literal

from dataset.openface_dataset import make_loaders
from model.train import run

# Config
ROOT = "dataset/UR_LYING_Deception_Dataset/splits"
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
SUBSAMPLE_K = 5
MOTION_METHOD: Literal["none", "feature_diff"] = "none"
MOTION_LOW = 0.2
MOTION_HIGH = float("inf")

train_loader, val_loader, test_loader, d_in = make_loaders(
    root_dir=ROOT,
    val_frac=0.2,
    batch_size=BATCH_SIZE,
    subsample_k=SUBSAMPLE_K,
    motion_method=MOTION_METHOD,
    motion_low=MOTION_LOW,
    motion_high=MOTION_HIGH,
)

print(
    f"Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}  |  Test: {len(test_loader.dataset)}  |  d_in: {d_in}\n"
)

run(
    train_loader=train_loader,
    val_loader=val_loader,
    d_in=d_in,
    device="cuda",
    epochs=EPOCHS,
    lr=LR,
    hidden=64,
    num_layers=1,
    dropout=0.4,
    save_path="best_bilstm.pt",
    scheduler="cosine",
)
