"""
Evaluate a saved BiLSTM or BiGRU checkpoint on the test split.

Usage:
    python test_model.py --ckpt best_urlying_bilstm.pt --root Openface_urlying --model bilstm
    python test_model.py --ckpt best_urlying_bigru.pt  --root Openface_urlying --model bigru
"""
import argparse

import torch
from torch.utils.data import DataLoader

from dataset.openface_dataset import OpenFaceDataset, _collate, DEFAULT_FEATURE_COLS
from model.BiLSTM import BiLSTMClassifier
from model.BiGRU import BiGRUClassifier


def evaluate(ckpt_path: str, root: str = "Openface_urlying", device_str: str = "mps",
             model_type: str = "bilstm", pooling: str = "attention"):
    device = torch.device(device_str if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    print(f"Loaded  : {ckpt_path}")
    print(f"Val acc : {ckpt['val_acc']:.4f}  (epoch {ckpt['epoch']})")

    d_in = len(DEFAULT_FEATURE_COLS)  # 48
    if model_type == "bilstm":
        model = BiLSTMClassifier(d_in=d_in, hidden=64, num_layers=1, dropout=0.0, pooling=pooling)
    elif model_type == "bigru":
        model = BiGRUClassifier(d_in=d_in, hidden=64, num_layers=1, dropout=0.0, pooling=pooling)
    else:
        raise ValueError(f"model_type must be 'bilstm' or 'bigru', got '{model_type}'")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    test_ds = OpenFaceDataset(root, split="Test")
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=_collate)

    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, lengths, y in test_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    acc = total_correct / total
    avg_loss = total_loss / total

    # Per-class breakdown
    tp = sum(p == 1 and l == 1 for p, l in zip(all_preds, all_labels))
    tn = sum(p == 0 and l == 0 for p, l in zip(all_preds, all_labels))
    fp = sum(p == 1 and l == 0 for p, l in zip(all_preds, all_labels))
    fn = sum(p == 0 and l == 1 for p, l in zip(all_preds, all_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*40}")
    print(f"Test samples : {total}")
    print(f"Test loss    : {avg_loss:.4f}")
    print(f"Test accuracy: {acc:.4f}  ({total_correct}/{total})")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 score     : {f1:.4f}")
    print(f"{'='*40}")
    print(f"Confusion matrix  (rows=actual, cols=predicted)")
    print(f"              Pred-T  Pred-D")
    print(f"  Actual-T :   {tn:4d}    {fp:4d}")
    print(f"  Actual-D :   {fn:4d}    {tp:4d}")
    print(f"{'='*40}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="best_urlying_bilstm.pt")
    parser.add_argument("--root", default="Openface_urlying")
    parser.add_argument("--model", default="bilstm", choices=["bilstm", "bigru"])
    parser.add_argument("--pooling", default="attention", choices=["attention", "mean", "last"])
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    evaluate(args.ckpt, args.root, args.device, args.model, args.pooling)
