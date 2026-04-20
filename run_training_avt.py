"""
Entry point for training the Late-Fusion BiGRU deception classifier
using visual (OpenFace) + audio (OpenSMILE) + text (Whisper/RoBERTa) features.

Usage:
    uv run python run_training_avt.py
"""

import matplotlib

matplotlib.use("Agg")

import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset.multimodal_dataset import (
    MultimodalDatasetAVT,
    _avt_collate,
    make_avt_loaders,
    openface_ur_lying_key,
    opensmile_ur_lying_key,
)
from model.LateFusionBiGRU import LateFusionBiGRUClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENFACE_ROOT = "dataset/UR_LYING_Deception_Dataset/openface_raw"
OPENSMILE_ROOT = "dataset/UR_LYING_Deception_Dataset/opensmile_raw"
WHISPER_ROOT = "dataset/UR_LYING_Deception_Dataset/whisper_raw"

BATCH_SIZE = 16
EPOCHS = 30
LR = 5e-4
HIDDEN = 128
NUM_LAYERS = 1
DROPOUT = 0.3
POOLING = "attention"
FUSION_HIDDEN = 128
PATIENCE = 5
SAVE_PATH = "best_bigru_avt.pt"
DEVICE = "cuda"
SEED = 42

AUDIO_SUBSAMPLE_K = 10
VISUAL_SUBSAMPLE_K = 1
AUDIO_MOTION_METHOD = "feature_diff"  # "none" | "feature_diff"
AUDIO_MOTION_LOW = 0.2
AUDIO_MOTION_HIGH = float("inf")
VISUAL_MOTION_METHOD = "feature_diff"  # "none" | "feature_diff"
VISUAL_MOTION_LOW = 0.2
VISUAL_MOTION_HIGH = float("inf")

USE_CV = False  # set True to run K-fold stratified CV instead of single train/val split
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for visual_x, visual_len, audio_x, audio_len, text_x, text_len, y in tqdm(
        loader, desc="Train", unit="batch"
    ):
        visual_x, visual_len = visual_x.to(device), visual_len.to(device)
        audio_x, audio_len = audio_x.to(device), audio_len.to(device)
        text_x, text_len = text_x.to(device), text_len.to(device)
        y = y.to(device)

        logits = model(visual_x, visual_len, audio_x, audio_len, text_x, text_len)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        correct += ((torch.sigmoid(logits) >= 0.5).float() == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Val",
) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for visual_x, visual_len, audio_x, audio_len, text_x, text_len, y in tqdm(
        loader, desc=desc, unit="batch"
    ):
        visual_x, visual_len = visual_x.to(device), visual_len.to(device)
        audio_x, audio_len = audio_x.to(device), audio_len.to(device)
        text_x, text_len = text_x.to(device), text_len.to(device)
        y = y.to(device)

        logits = model(visual_x, visual_len, audio_x, audio_len, text_x, text_len)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        correct += ((torch.sigmoid(logits) >= 0.5).float() == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


def _make_model(
    visual_d_in: int, audio_d_in: int, text_d_in: int, device: torch.device
) -> LateFusionBiGRUClassifier:
    return LateFusionBiGRUClassifier(
        visual_d_in=visual_d_in,
        audio_d_in=audio_d_in,
        text_d_in=text_d_in,
        hidden=HIDDEN,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pooling=POOLING,
        fusion_hidden=FUSION_HIDDEN,
        use_visual=True,
        use_audio=True,
        use_text=True,
    ).to(device)


def _loader_kwargs(
    batch_size: int = BATCH_SIZE,
    *,
    seed: int | None = None,
) -> dict:
    kwargs: dict = dict(batch_size=batch_size, collate_fn=_avt_collate, pin_memory=True)
    if seed is not None:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return kwargs


# ---------------------------------------------------------------------------
# Dataset kwargs shared by both modes
# ---------------------------------------------------------------------------
_ds_kwargs: dict = dict(
    visual_key_fn=openface_ur_lying_key,
    audio_key_fn=opensmile_ur_lying_key,
    audio_subsample_k=AUDIO_SUBSAMPLE_K,
    visual_subsample_k=VISUAL_SUBSAMPLE_K,
    audio_motion_method=AUDIO_MOTION_METHOD,
    audio_motion_low=AUDIO_MOTION_LOW,
    audio_motion_high=AUDIO_MOTION_HIGH,
    visual_motion_method=VISUAL_MOTION_METHOD,
    visual_motion_low=VISUAL_MOTION_LOW,
    visual_motion_high=VISUAL_MOTION_HIGH,
)

seed_everything(SEED)
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Mode A: single train / val / test split
# ---------------------------------------------------------------------------
if not USE_CV:
    train_loader, val_loader, test_loader, visual_d_in, audio_d_in, text_d_in = (
        make_avt_loaders(
            openface_root=OPENFACE_ROOT,
            opensmile_root=OPENSMILE_ROOT,
            whisper_root=WHISPER_ROOT,
            val_frac=0.2,
            batch_size=BATCH_SIZE,
            seed=SEED,
            **_ds_kwargs,
        )
    )

    print(
        f"Train: {len(train_loader.dataset)}"
        f"  |  Val: {len(val_loader.dataset)}"
        f"  |  Test: {len(test_loader.dataset)}"
        f"  |  visual_d_in: {visual_d_in}"
        f"  |  audio_d_in: {audio_d_in}"
        f"  |  text_d_in: {text_d_in}\n"
    )

    model = _make_model(visual_d_in, audio_d_in, text_d_in, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    no_improve = 0

    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    print("Start training...\n")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(
            train_loader, model, optimizer, criterion, device
        )
        vl_loss, vl_acc = eval_one_epoch(val_loader, model, criterion, device)

        train_losses.append(tr_loss)
        train_accs.append(tr_acc)
        val_losses.append(vl_loss)
        val_accs.append(vl_acc)

        print(
            f"Epoch {epoch}/{EPOCHS}"
            f" | Train Loss {tr_loss:.4f} Acc {tr_acc:.4f}"
            f" | Val Loss {vl_loss:.4f} Acc {vl_acc:.4f}"
        )

        if vl_loss < best_val_loss or vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_val_loss = vl_loss
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "model_type": "multimodal_avt",
                    "model_config": {
                        "visual_d_in": visual_d_in,
                        "audio_d_in": audio_d_in,
                        "text_d_in": text_d_in,
                        "hidden": HIDDEN,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "pooling": POOLING,
                        "fusion_hidden": FUSION_HIDDEN,
                    },
                    "dataset_config": {
                        "openface_root": OPENFACE_ROOT,
                        "opensmile_root": OPENSMILE_ROOT,
                        "whisper_root": WHISPER_ROOT,
                        "audio_subsample_k": AUDIO_SUBSAMPLE_K,
                        "visual_subsample_k": VISUAL_SUBSAMPLE_K,
                        "audio_motion_method": AUDIO_MOTION_METHOD,
                        "audio_motion_low": AUDIO_MOTION_LOW,
                        "audio_motion_high": AUDIO_MOTION_HIGH,
                        "visual_motion_method": VISUAL_MOTION_METHOD,
                        "visual_motion_low": VISUAL_MOTION_LOW,
                        "visual_motion_high": VISUAL_MOTION_HIGH,
                    },
                },
                SAVE_PATH,
            )
            print(
                f"  → saved best model, best_val_loss={best_val_loss:.4f}, best_val_acc={best_val_acc:.4f}\n"
            )
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}.\n")
                break

    print(f"\nTraining finished. Best Val Acc = {best_val_acc:.4f}")

    # Test set evaluation
    test_loss, test_acc = eval_one_epoch(
        test_loader, model, criterion, device, desc="Test"
    )
    print(f"Test  Loss {test_loss:.4f} | Test  Acc {test_acc:.4f}")

    # Plots
    for name, train_vals, val_vals, ylabel in [
        ("loss_curve_avt.png", train_losses, val_losses, "Loss"),
        ("accuracy_curve_avt.png", train_accs, val_accs, "Accuracy"),
    ]:
        plt.figure()
        plt.plot(train_vals, label=f"Train {ylabel}")
        plt.plot(val_vals, label=f"Val {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Training and Validation {ylabel}")
        plt.legend()
        plt.grid()
        plt.savefig(name, dpi=150, bbox_inches="tight")
        plt.close()

# ---------------------------------------------------------------------------
# Mode B: stratified K-fold cross-validation
# ---------------------------------------------------------------------------
else:
    base_ds = MultimodalDatasetAVT(
        OPENFACE_ROOT, OPENSMILE_ROOT, WHISPER_ROOT, "Train", **_ds_kwargs
    )
    visual_d_in = base_ds.visual_d_in
    audio_d_in = base_ds.audio_d_in
    text_d_in = base_ds.text_d_in
    labels = [lbl for _, _, _, lbl in base_ds.samples]

    test_ds = MultimodalDatasetAVT(
        OPENFACE_ROOT, OPENSMILE_ROOT, WHISPER_ROOT, "Test", **_ds_kwargs
    )
    test_loader = DataLoader(test_ds, shuffle=False, **_loader_kwargs(seed=SEED))

    print(
        f"Train: {len(base_ds)}"
        f"  |  Test: {len(test_ds)}"
        f"  |  visual_d_in: {visual_d_in}"
        f"  |  audio_d_in: {audio_d_in}"
        f"  |  text_d_in: {text_d_in}"
        f"  |  folds: {CV_FOLDS}\n"
    )

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    splits = list(skf.split(np.arange(len(base_ds)), labels))

    fold_accs: list[float] = []
    fold_best_epochs: list[int] = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"{'=' * 55}")
        print(f"Fold {fold}/{CV_FOLDS}  |  train={len(train_idx)}  val={len(val_idx)}")
        print(f"{'=' * 55}")

        fold_train_loader = DataLoader(
            Subset(base_ds, train_idx),
            shuffle=True,
            **_loader_kwargs(seed=SEED + fold),
        )
        fold_val_loader = DataLoader(
            Subset(base_ds, val_idx),
            shuffle=False,
            **_loader_kwargs(seed=SEED + CV_FOLDS + fold),
        )

        fold_model = _make_model(visual_d_in, audio_d_in, text_d_in, device)
        fold_criterion = nn.BCEWithLogitsLoss()
        fold_optimizer = optim.AdamW(fold_model.parameters(), lr=LR, weight_decay=1e-4)

        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_epoch = 1
        no_improve = 0

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(
                fold_train_loader, fold_model, fold_optimizer, fold_criterion, device
            )
            vl_loss, vl_acc = eval_one_epoch(
                fold_val_loader, fold_model, fold_criterion, device
            )

            print(
                f"  Epoch {epoch}/{EPOCHS}"
                f" | Train Loss {tr_loss:.4f} Acc {tr_acc:.4f}"
                f" | Val Loss {vl_loss:.4f} Acc {vl_acc:.4f}"
            )

            if vl_acc > best_val_acc:
                best_val_acc = vl_acc
                best_epoch = epoch

            if vl_loss < best_val_loss:
                best_val_loss = vl_loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch}.\n")
                    break

        fold_accs.append(best_val_acc)
        fold_best_epochs.append(best_epoch)
        print(f"  Fold {fold} best val acc: {best_val_acc:.4f}  (epoch {best_epoch})\n")

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    avg_epoch = int(np.ceil(np.mean(fold_best_epochs)))

    print(f"{'=' * 55}")
    print(f"CV Results ({CV_FOLDS} folds):")
    for i, acc in enumerate(fold_accs, 1):
        print(f"  Fold {i}: {acc:.4f}")
    print(f"  Mean val acc : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Avg best epoch: {avg_epoch}")
    print(f"{'=' * 55}\n")

    # ── Final model: train on all Train data for avg_best_epoch epochs ──────
    print(
        f"Training final model on all {len(base_ds)} train samples for {avg_epoch} epoch(s)...\n"
    )

    full_train_loader = DataLoader(
        base_ds, shuffle=True, **_loader_kwargs(seed=SEED + 2 * CV_FOLDS + 1)
    )
    model = _make_model(visual_d_in, audio_d_in, text_d_in, device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    final_train_losses: list[float] = []
    final_train_accs: list[float] = []

    for epoch in range(1, avg_epoch + 1):
        tr_loss, tr_acc = train_one_epoch(
            full_train_loader, model, optimizer, criterion, device
        )
        final_train_losses.append(tr_loss)
        final_train_accs.append(tr_acc)
        print(f"Epoch {epoch}/{avg_epoch} | Train Loss {tr_loss:.4f} Acc {tr_acc:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "mean_val_acc": mean_acc,
            "avg_epoch": avg_epoch,
            "model_type": "multimodal_avt",
            "model_config": {
                "visual_d_in": visual_d_in,
                "audio_d_in": audio_d_in,
                "text_d_in": text_d_in,
                "hidden": HIDDEN,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "pooling": POOLING,
                "fusion_hidden": FUSION_HIDDEN,
            },
            "dataset_config": {
                "openface_root": OPENFACE_ROOT,
                "opensmile_root": OPENSMILE_ROOT,
                "whisper_root": WHISPER_ROOT,
                "audio_subsample_k": AUDIO_SUBSAMPLE_K,
                "visual_subsample_k": VISUAL_SUBSAMPLE_K,
                "audio_motion_method": AUDIO_MOTION_METHOD,
                "audio_motion_low": AUDIO_MOTION_LOW,
                "audio_motion_high": AUDIO_MOTION_HIGH,
                "visual_motion_method": VISUAL_MOTION_METHOD,
                "visual_motion_low": VISUAL_MOTION_LOW,
                "visual_motion_high": VISUAL_MOTION_HIGH,
            },
        },
        SAVE_PATH,
    )
    print(f"\nSaved final model → {SAVE_PATH}")

    # ── Test evaluation with full metrics ────────────────────────────────────
    print(f"\n{'=' * 55}")
    print("Evaluating on test set...")
    print(f"{'=' * 55}")

    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds: list[float] = []
    all_labels_list: list[float] = []

    with torch.no_grad():
        for visual_x, visual_len, audio_x, audio_len, text_x, text_len, y in tqdm(
            test_loader, desc="Test", unit="batch"
        ):
            visual_x, visual_len = visual_x.to(device), visual_len.to(device)
            audio_x, audio_len = audio_x.to(device), audio_len.to(device)
            text_x, text_len = text_x.to(device), text_len.to(device)
            y = y.to(device)

            logits = model(visual_x, visual_len, audio_x, audio_len, text_x, text_len)
            loss = criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels_list.extend(y.cpu().tolist())

    acc = total_correct / total
    avg_loss = total_loss / total

    tp = sum(p == 1 and lbl == 1 for p, lbl in zip(all_preds, all_labels_list))
    tn = sum(p == 0 and lbl == 0 for p, lbl in zip(all_preds, all_labels_list))
    fp = sum(p == 1 and lbl == 0 for p, lbl in zip(all_preds, all_labels_list))
    fn = sum(p == 0 and lbl == 1 for p, lbl in zip(all_preds, all_labels_list))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"\nTest samples : {total}")
    print(f"Test loss    : {avg_loss:.4f}")
    print(f"Test accuracy: {acc:.4f}  ({total_correct}/{total})")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 score     : {f1:.4f}")
    print("\nConfusion matrix  (rows=actual, cols=predicted)")
    print("              Pred-T  Pred-D")
    print(f"  Actual-T :   {tn:4d}    {fp:4d}")
    print(f"  Actual-D :   {fn:4d}    {tp:4d}")
    print(f"{'=' * 55}")

    # Plot final model training curve
    for name, vals, ylabel in [
        ("loss_curve_avt.png", final_train_losses, "Loss"),
        ("accuracy_curve_avt.png", final_train_accs, "Accuracy"),
    ]:
        plt.figure()
        plt.plot(vals, label=f"Train {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"Final Model Training {ylabel} (CV mode)")
        plt.legend()
        plt.grid()
        plt.savefig(name, dpi=150, bbox_inches="tight")
        plt.close()
