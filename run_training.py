"""
Entry point for training the BiLSTM deception classifier.

Usage:
    uv run python run_training.py
"""
import matplotlib
matplotlib.use("Agg")

from dataset.openface_dataset import make_loaders
from model.train import run

# Config
ROOT       = "OpenFace_features"
BATCH_SIZE = 16
EPOCHS     = 20
LR         = 1e-3

train_loader, val_loader, test_loader, d_in = make_loaders(
    root_dir=ROOT,
    val_frac=0.2,
    batch_size=BATCH_SIZE,
)

print(f"Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}  |  Test: {len(test_loader.dataset)}  |  d_in: {d_in}\n")

run(
    train_loader=train_loader,
    val_loader=val_loader,
    d_in=d_in,
    device="mps",
    epochs=EPOCHS,
    lr=LR,
    hidden=64,
    num_layers=1,
    dropout=0.4,
    save_path="best_bilstm.pt",
)
