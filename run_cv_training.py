"""
CV hyperparameter search + final model training + test evaluation.

Steps:
  1. Grid search over (hidden, dropout, lr) using 5-fold stratified CV on Train/
  2. Train a final model on all 109 Train/ samples with the best hyperparameters
  3. Evaluate once on the 12-sample Test/ set

Usage:
    uv run python run_cv_training.py
"""
import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from dataset.openface_dataset import OpenFaceDataset, _collate, DEFAULT_FEATURE_COLS
from model.train import run, train_one_epoch
from model.BiLSTM import BiLSTMClassifier

# ── Config ────────────────────────────────────────────────────────────────────
ROOT       = "dataset/UR_LYING_Deception_Dataset/splits"
K           = 5
BATCH_SIZE  = 16
EPOCHS      = 20
PATIENCE    = 10
DEVICE      = "cuda"
SUBSAMPLE_K = 5
SCHEDULER   = "cosine"
D_IN        = len(DEFAULT_FEATURE_COLS)  # 48

# Hyperparameter grid  (2 × 3 × 2 = 12 combos)
PARAM_GRID = [
    {"hidden": 64, "dropout": 0.4, "lr": lr}
    for lr in [4e-4, 6e-4, 8e-4, 1e-3, 2e-3, 3e-3]
]

# ── Datasets ──────────────────────────────────────────────────────────────────
full_train_ds = OpenFaceDataset(ROOT, split="Train", subsample_k=SUBSAMPLE_K)
test_ds       = OpenFaceDataset(ROOT, split="Test",  subsample_k=SUBSAMPLE_K)
labels        = [lbl for _, lbl in full_train_ds.samples]

print(f"Train: {len(full_train_ds)}  |  Test: {len(test_ds)}  |  d_in: {D_IN}")
print(f"Grid: {len(PARAM_GRID)} combos × {K} folds = {len(PARAM_GRID) * K} runs\n")

# ── CV hyperparameter search ──────────────────────────────────────────────────
skf    = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
splits = list(skf.split(np.arange(len(full_train_ds)), labels))

results = []

for i, params in enumerate(PARAM_GRID):
    fold_accs   = []
    fold_epochs = []

    for fold, (train_idx, val_idx) in enumerate(splits):
        train_loader = DataLoader(
            Subset(full_train_ds, train_idx),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate, pin_memory=False,
        )
        val_loader = DataLoader(
            Subset(full_train_ds, val_idx),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate, pin_memory=False,
        )

        tmp_path = f"_tmp_cv.pt"
        run(
            train_loader=train_loader,
            val_loader=val_loader,
            d_in=D_IN,
            device=DEVICE,
            epochs=EPOCHS,
            lr=params["lr"],
            hidden=params["hidden"],
            num_layers=1,
            dropout=params["dropout"],
            patience=PATIENCE,
            save_path=tmp_path,
            verbose=True,
            scheduler=SCHEDULER,
        )

        ckpt = torch.load(tmp_path, weights_only=True)
        fold_accs.append(ckpt["val_acc"])
        fold_epochs.append(ckpt["epoch"])
        os.remove(tmp_path)

    mean_acc  = float(np.mean(fold_accs))
    avg_epoch = int(np.ceil(np.mean(fold_epochs)))
    results.append({**params, "mean_val_acc": mean_acc, "avg_best_epoch": avg_epoch})

    print(
        f"[{i+1:2d}/{len(PARAM_GRID)}] "
        f"hidden={params['hidden']:3d}  dropout={params['dropout']}  lr={params['lr']:.0e}"
        f"  →  mean_val_acc={mean_acc:.4f}  avg_best_epoch={avg_epoch}"
    )

# ── Select best hyperparameters ───────────────────────────────────────────────
best = max(results, key=lambda r: r["mean_val_acc"])

print(f"\n{'='*55}")
print(f"Best: hidden={best['hidden']}  dropout={best['dropout']}  lr={best['lr']:.0e}")
print(f"  CV mean val acc : {best['mean_val_acc']:.4f}")
print(f"  Avg best epoch  : {best['avg_best_epoch']}")
print(f"{'='*55}\n")

# ── Final training on all 109 samples ─────────────────────────────────────────
print("Training final model on all 109 training samples...")

device = torch.device(DEVICE if torch.backends.mps.is_available() else "cpu")

final_model = BiLSTMClassifier(
    d_in=D_IN,
    hidden=best["hidden"],
    num_layers=1,
    dropout=best["dropout"],
    pooling="mean",
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(final_model.parameters(), lr=best["lr"], weight_decay=1e-4)

# Train for avg_best_epoch + 1 epochs (epoch index is 0-based in checkpoint)
final_epochs = best["avg_best_epoch"] + 1
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs)

full_train_loader = DataLoader(
    full_train_ds,
    batch_size=BATCH_SIZE, shuffle=True, collate_fn=_collate, pin_memory=False,
)

for epoch in range(final_epochs):
    loss, acc = train_one_epoch(final_model, full_train_loader, optimizer, criterion, device)
    current_lr = optimizer.param_groups[0]["lr"]
    lr_scheduler.step()
    print(f"Epoch {epoch+1}/{final_epochs}  |  LR {current_lr:.2e}  |  Train Loss {loss:.4f}  |  Train Acc {acc:.4f}")

torch.save({"model_state_dict": final_model.state_dict(), "params": best}, "best_bilstm_final.pt")
print("Saved → best_bilstm_final.pt\n")

# ── Test evaluation ───────────────────────────────────────────────────────────
print(f"{'='*55}")
print("Evaluating on test set...")
print(f"{'='*55}")

test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=_collate)

final_model.eval()
total_loss, total_correct, total = 0.0, 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for x, lengths, y in test_loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = final_model(x, lengths)
        loss   = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == y).sum().item()
        total += x.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(y.cpu().tolist())

acc     = total_correct / total
avg_loss = total_loss / total

tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
tn = sum(p == 0 and l == 0 for p, l in zip(all_preds, all_labels))
fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\nTest samples : {total}")
print(f"Test loss    : {avg_loss:.4f}")
print(f"Test accuracy: {acc:.4f}  ({total_correct}/{total})")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 score     : {f1:.4f}")
print(f"\nConfusion matrix  (rows=actual, cols=predicted)")
print(f"              Pred-T  Pred-D")
print(f"  Actual-T :   {tn:4d}    {fp:4d}")
print(f"  Actual-D :   {fn:4d}    {tp:4d}")
print(f"{'='*55}")
