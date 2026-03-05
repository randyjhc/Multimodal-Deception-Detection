from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import BiLSTM
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
)
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd

def collate_fn(batch):
    '''
    batch: list of (seq(T_i, D), y, vid)
    '''
    seqs, ys, vids_key = zip(*batch)  # Need 
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    x_padded = pad_sequence(seqs, batch_first=True)  # pad with 0
    y = torch.tensor(ys, dtype=torch.float32)  # 0/1
    return x_padded, lengths, y, vids_key  # vids_key: tuple of video_id


def train_one_epoch(model, loader, optimizer, criterion, device="cuda"):
    model.train()
    #criterion = torch.nn.BCEWithLogitsLoss()

    total_loss, total_correct, total = 0.0, 0, 0

    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        logits = model(x, lengths)  # (B,)
        loss = criterion(logits, y)  # y: float 0/1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent gradient explosion
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return (
        total_loss / total,
        total_correct / total
    )


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device="cuda"):
    model.eval()

    total_loss, total_correct, total = 0.0, 0, 0

    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        logits = model(x, lengths)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_correct += (preds == y).sum().item()
        total += x.size(0)

    return (
        total_loss / total,
        total_correct / total
    )

def run(
        train_loader,
        val_loader,
        d_in,
        device="cuda",
        epochs=20,
        lr=1e-3,
        hidden=128,
        num_layers=1,
        dropout=0.2,
        save_path="best_bilstm.pt"
    ):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # model
    model = BiLSTM.BiLSTMClassifier(
        d_in=d_in,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
        pooling="mean"
    ).to(device)

    # loss
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0

    list_train_loss = []
    list_train_acc = []
    list_val_loss = []
    list_val_acc = []

    print("Start training...\n")

    for epoch in range(epochs):

        # ===== TRAIN =====
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        # ===== VALIDATION =====
        val_loss, val_acc = eval_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        list_train_loss.append(train_loss)
        list_train_acc.append(train_acc)
        list_val_loss.append(val_loss)
        list_val_acc.append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs}"
            f" | Train Loss {train_loss:.4f}"
            f" | Train Acc {train_acc:.4f}"
            f" | Val Loss {val_loss:.4f}"
            f" | Val Acc {val_acc:.4f}"
        )

        # save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch
                },
                save_path
            )
            print("Saved best model.\n")

    print(f"\nTraining finished. Best Val Acc = {best_val_acc:.4f}")

    # ===== Plot Loss =====
    plt.figure()

    plt.plot(list_train_loss, label="Train Loss")
    plt.plot(list_val_loss, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid()

    plt.show()

    ''' 
    """
    Not meaningful to measure accuracy for window level
    """
    # ===== Plot Accuracy =====
    plt.figure()

    plt.plot(list_train_acc, label="Train Accuracy")
    plt.plot(list_val_acc, label="Val Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.legend()
    plt.grid()

    plt.show()
    '''

    return model


@torch.no_grad()
def test(model, loader, criterion, device="cuda"):
    '''
    Works if the dataset is the whole video sequence
    '''

    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    total_loss = 0
    total = 0

    for x, lengths, y in loader:

        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        logits = model(x, lengths)

        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        total += x.size(0)

    avg_loss = total_loss / total

    # ===== F1 score =====
    f1 = f1_score(all_labels, all_preds)

    # ===== Confusion Matrix =====
    cm = confusion_matrix(all_labels, all_preds)

    print("Confusion Matrix")
    print(cm)

    # ===== ROC Curve =====
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        "loss": avg_loss,
        "f1": f1,
        "confusion_matrix": cm,
        "auc": roc_auc
    }
    
@torch.no_grad()
def evaluate_video_level(model, loader, criterion, device="cuda",
                         agg="mean", topk=0.2, threshold=0.5, plot_roc=True):
    '''
    Video-level-evaluation
    '''

    model.eval()

    # Initialize 2 dictionaries for video-level evaluation 
    vid2probs = defaultdict(list) # Dictionary of video to list of probabilities
    vid2label = {} # Dictionary of video to label

    total_loss = 0.0
    total_windows = 0

    for x, lengths, y, vids in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)

        logits = model(x, lengths)          # (B,)
        loss = criterion(logits, y)

        probs = torch.sigmoid(logits).detach().cpu().numpy()  # (B,)
        y_np  = y.detach().cpu().numpy()                      # (B,)

        total_loss += loss.item() * x.size(0)
        total_windows += x.size(0)

        # Process and classify each test data probability to the corresponding video
        for p, lab, vid in zip(probs, y_np, vids):
            vid2probs[vid].append(float(p))
            
            if vid not in vid2label:
                vid2label[vid] = int(lab)
            else:
                assert vid2label[vid] == int(lab)  # Check video's label consistency
                pass

    avg_window_loss = total_loss / max(1, total_windows)

    # ===== Aggregation：window -> video probability =====
    y_true = []
    y_prob = []

    for vid, probs_list in vid2probs.items():
        probs_arr = np.array(probs_list, dtype=np.float32)

        if agg == "mean":
            p_vid = float(probs_arr.mean())
        elif agg == "max":
            p_vid = float(probs_arr.max())
        elif agg == "topk":  # Highly suggested
            k = max(1, int(len(probs_arr) * topk))
            top = np.sort(probs_arr)[-k:]  # Sort and get last [-k:] part
            p_vid = float(top.mean())
        else:
            raise ValueError(f"Unknown agg={agg}. Use mean/max/topk.")

        y_true.append(vid2label[vid])
        y_prob.append(p_vid)

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(int)

    # ===== metrics (video-level) =====
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)

    # ROC/AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"[Video-level] agg={agg} | videos={len(y_true)} | window_loss={avg_window_loss:.4f}")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)

    if plot_roc:
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Video-level ROC Curve")
        plt.legend()
        plt.grid()
        plt.show()

    return {
        "window_loss": avg_window_loss,
        "video_acc": acc,
        "video_f1": f1,
        "video_cm": cm,
        "video_auc": roc_auc,
        "y_true": y_true,
        "y_prob": y_prob,
    }