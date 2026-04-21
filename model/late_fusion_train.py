from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from model import LateFusionBiGRU
import torch.optim as optim
import matplotlib.pyplot as plt

def collate_fn(batch):
    visual_seqs, audio_seqs, text_seqs, ys = zip(*batch)
    visual_lengths = torch.tensor([v.shape[0] for v in visual_seqs], dtype=torch.long)
    audio_lengths  = torch.tensor([a.shape[0] for a in audio_seqs], dtype=torch.long)
    text_lengths   = torch.tensor([t.shape[0] for t in text_seqs], dtype=torch.long)
    
    visual_x = pad_sequence(visual_seqs, batch_first=True)
    audio_x  = pad_sequence(audio_seqs, batch_first=True)
    text_x   = pad_sequence(text_seqs, batch_first=True)

    y = torch.tensor(ys, dtype=torch.float32)

    return visual_x, visual_lengths, audio_x, audio_lengths, text_x, text_lengths, y


def train_one_epoch(model, loader, optimizer, criterion, device="cuda"):
    model.train()
    #criterion = torch.nn.BCEWithLogitsLoss()

    total_loss, total_correct, total = 0.0, 0, 0

    for visual_x, visual_lengths, audio_x, audio_lengths, text_x, text_lengths, y in loader:
        visual_x = visual_x.to(device)
        visual_lengths = visual_lengths.to(device)
        audio_x = audio_x.to(device)
        audio_lengths = audio_lengths.to(device)
        text_x = text_x.to(device)
        text_lengths = text_lengths.to(device)
        y = y.to(device)

        logits = model(
            visual_x, visual_lengths,
            audio_x, audio_lengths,
            text_x, text_lengths
        )
        loss = criterion(logits, y)  # y: float 0/1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent gradient explosion
        optimizer.step()

        total_loss += loss.item() * y.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_correct += (preds == y).sum().item()
        total += y.size(0)

    return (
        total_loss / total,
        total_correct / total
    )


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device="cuda"):
    model.eval()

    total_loss, total_correct, total = 0.0, 0, 0

    for visual_x, visual_lengths, audio_x, audio_lengths, text_x, text_lengths, y in loader:
        visual_x = visual_x.to(device)
        visual_lengths = visual_lengths.to(device)
        audio_x = audio_x.to(device)
        audio_lengths = audio_lengths.to(device)
        text_x = text_x.to(device)
        text_lengths = text_lengths.to(device)
        y = y.to(device)

        logits = model(
            visual_x, visual_lengths,
            audio_x, audio_lengths,
            text_x, text_lengths
        )
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        total_correct += (preds == y).sum().item()
        total += y.size(0)

    return (
        total_loss / total,
        total_correct / total
    )

def run(
    train_loader,
    val_loader,
    visual_d_in,
    audio_d_in,
    text_d_in,
    u_visual, # Use Visual encoder
    u_audio,  # Use Audio encoder
    u_text,   # Use Text encoder
    pooling="attention",   # "mean" | "max" | "last" | "topk_mean" | "attn"
    device="cuda",
    epochs=20,
    lr=1e-3,
    hidden=128,
    f_hidden=128,
    num_layers=1,
    dropout=0.2,
    save_path="best_bilstm.pt",
    patience=5,
    verbose=True,
):

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = LateFusionBiGRU.LateFusionBiGRUClassifier(
        visual_d_in=visual_d_in,
        audio_d_in=audio_d_in,
        text_d_in=text_d_in,
        hidden=hidden,
        num_layers=num_layers,
        dropout=dropout,
        pooling=pooling,
        top_k=5,
        fusion_hidden=f_hidden,
        use_visual=u_visual,
        use_audio=u_audio,
        use_text=u_text
    ).to(device)

    # loss
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    epochs_no_improve = 0

    list_train_loss = []
    list_train_acc = []
    list_val_loss = []
    list_val_acc = []

    if verbose:
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

        if verbose:
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
            if verbose:
                print("Saved best model.\n")

        # early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch+1} (no val loss improvement for {patience} epochs).\n")
                break

    if verbose:
        print(f"\nTraining finished. Best Val Acc = {best_val_acc:.4f}")

    if not verbose:
        return model

    # ===== Plot Loss =====
    plt.figure()

    plt.plot(list_train_loss, label="Train Loss")
    plt.plot(list_val_loss, label="Val Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid()

    plt.savefig("loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ===== Plot Accuracy =====
    plt.figure()

    plt.plot(list_train_acc, label="Train Accuracy")
    plt.plot(list_val_acc, label="Val Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")

    plt.legend()
    plt.grid()

    plt.savefig("accuracy_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    return model