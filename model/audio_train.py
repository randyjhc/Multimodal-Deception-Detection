import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from model.BiGRU import BiGRUClassifier
from model.BiLSTM import BiLSTMClassifier


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, total_correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0

    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / total, total_correct / total


def evaluate_on_test(
    checkpoint_path: str,
    test_loader,
    d_in: int,
    model_type: str = "bigru",
    device: str = "mps",
    hidden: int = 64,
    num_layers: int = 1,
    dropout: float = 0.0,
    pooling: str = "attention",
):
    device = torch.device(device if (torch.cuda.is_available() or device == "mps") else "cpu")

    if model_type == "bigru":
        model = BiGRUClassifier(
            d_in=d_in, hidden=hidden, num_layers=num_layers,
            dropout=dropout, pooling=pooling,
        ).to(device)
    elif model_type == "bilstm":
        model = BiLSTMClassifier(
            d_in=d_in, hidden=hidden, num_layers=num_layers,
            dropout=dropout, pooling=pooling,
        ).to(device)
    else:
        raise ValueError(f"model_type must be 'bigru' or 'bilstm', got '{model_type}'")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    test_loss, test_acc = eval_one_epoch(model, test_loader, criterion, device)
    return {"test_loss": test_loss, "test_acc": test_acc}


def run(
    train_loader,
    val_loader,
    d_in: int,
    model_type: str = "bigru",   # "bigru" or "bilstm"
    device: str = "mps",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: int = 64,
    num_layers: int = 1,
    dropout: float = 0.4,
    pooling: str = "attention",
    save_path: str = "best_frame.pt",
    patience: int = 10,
    verbose: bool = True,
    plot_prefix: str = "frame",
):
    device = torch.device(device if (torch.cuda.is_available() or device == "mps") else "cpu")

    if model_type == "bigru":
        model = BiGRUClassifier(
            d_in=d_in, hidden=hidden, num_layers=num_layers,
            dropout=dropout, pooling=pooling,
        ).to(device)
    elif model_type == "bilstm":
        model = BiLSTMClassifier(
            d_in=d_in, hidden=hidden, num_layers=num_layers,
            dropout=dropout, pooling=pooling,
        ).to(device)
    else:
        raise ValueError(f"model_type must be 'bigru' or 'bilstm', got '{model_type}'")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Training {model_type.upper()} (frame-level LLD, d_in={d_in})")
        print(f"{'='*60}")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"Epoch {epoch+1:>3}/{epochs}"
                f" | Train Loss {train_loss:.4f}  Acc {train_acc:.4f}"
                f" | Val Loss {val_loss:.4f}  Acc {val_acc:.4f}"
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state_dict": model.state_dict(), "val_acc": val_acc, "epoch": epoch},
                save_path,
            )
            if verbose:
                print(f"  -> Saved best model (val_acc={val_acc:.4f})\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}.\n")
                break

    if verbose:
        print(f"\nBest Val Acc ({model_type.upper()}): {best_val_acc:.4f}\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_title(f"{model_type.upper()} — Loss (frame-level)")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_title(f"{model_type.upper()} — Accuracy (frame-level)")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid()

    fig.tight_layout()
    fig.savefig(f"{plot_prefix}_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"model": model, "best_val_acc": best_val_acc, "history": history}
