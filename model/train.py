from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import BiLSTM
import torch.optim as optim
import matplotlib as plt

def collate_fn(batch):
    """
    batch: list of (seq(T_i, D), y)
    returns:
      x_padded: (B, T_max, D)
      lengths: (B,)
      y: (B,)
    """
    seqs, ys = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    x_padded = pad_sequence(seqs, batch_first=True)  # pad with 0
    y = torch.tensor(ys, dtype=torch.float32)  # 0/1
    return x_padded, lengths, y


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

    return model