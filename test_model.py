"""
Evaluate a saved checkpoint on the test split.

The checkpoint stores all model and dataset configuration, so no architecture
flags are needed — just point to the checkpoint file.

Usage:
    uv run python test_model.py --ckpt best_bigru.pt
    uv run python test_model.py --ckpt best_bilstm.pt --root OpenFace_features
    uv run python test_model.py --ckpt best_bigru.pt \\
        --root /alt/openface --opensmile_root /alt/opensmile --device cuda
"""

import argparse

import torch
from torch.utils.data import DataLoader

from dataset.multimodal_dataset import (
    MultimodalDataset,
    MultimodalDatasetAVT,
    _avt_collate,
    _multimodal_collate,
    openface_ur_lying_key,
    opensmile_ur_lying_key,
)
from dataset.openface_dataset import DEFAULT_FEATURE_COLS, OpenFaceDataset, _collate
from model.BiLSTM import BiLSTMClassifier
from model.LateFusionBiGRU import LateFusionBiGRUClassifier


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _print_metrics(
    total: int,
    avg_loss: float,
    total_correct: int,
    all_preds: list[float],
    all_labels: list[float],
) -> None:
    tp = sum(p == 1 and lbl == 1 for p, lbl in zip(all_preds, all_labels))
    tn = sum(p == 0 and lbl == 0 for p, lbl in zip(all_preds, all_labels))
    fp = sum(p == 1 and lbl == 0 for p, lbl in zip(all_preds, all_labels))
    fn = sum(p == 0 and lbl == 1 for p, lbl in zip(all_preds, all_labels))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print(f"\n{'=' * 40}")
    print(f"Test samples : {total}")
    print(f"Test loss    : {avg_loss:.4f}")
    print(f"Test accuracy: {total_correct / total:.4f}  ({total_correct}/{total})")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 score     : {f1:.4f}")
    print(f"{'=' * 40}")
    print("Confusion matrix  (rows=actual, cols=predicted)")
    print("              Pred-T  Pred-D")
    print(f"  Actual-T :   {tn:4d}    {fp:4d}")
    print(f"  Actual-D :   {fn:4d}    {tp:4d}")
    print(f"{'=' * 40}")


def evaluate_bilstm(
    ckpt: dict,
    root: str | None,
    device: torch.device,
) -> None:
    mc = ckpt.get("model_config", {})
    dc = ckpt.get("dataset_config", {})
    openface_root = root or dc.get("openface_root", "OpenFace_features")

    d_in = len(DEFAULT_FEATURE_COLS)
    model = BiLSTMClassifier(
        d_in=d_in,
        hidden=mc.get("hidden", 64),
        num_layers=mc.get("num_layers", 1),
        dropout=mc.get("dropout", 0.4),
        pooling=mc.get("pooling", "mean"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    test_ds = OpenFaceDataset(openface_root, split="Test")
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=_collate)

    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for x, lengths, y in test_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            logits = model(x, lengths)
            total_loss += criterion(logits, y).item() * x.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total += x.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    _print_metrics(total, total_loss / total, total_correct, all_preds, all_labels)


def evaluate_multimodal(
    ckpt: dict,
    root: str | None,
    opensmile_root: str | None,
    device: torch.device,
) -> None:
    mc = ckpt["model_config"]
    dc = ckpt["dataset_config"]
    openface_root = root or dc["openface_root"]
    audio_root = opensmile_root or dc["opensmile_root"]

    test_ds = MultimodalDataset(
        openface_root,
        audio_root,
        split="Test",
        visual_key_fn=openface_ur_lying_key,
        audio_key_fn=opensmile_ur_lying_key,
        audio_subsample_k=dc["audio_subsample_k"],
        visual_subsample_k=dc["visual_subsample_k"],
        audio_motion_method=dc["audio_motion_method"],
        audio_motion_low=dc["audio_motion_low"],
        audio_motion_high=dc["audio_motion_high"],
        visual_motion_method=dc["visual_motion_method"],
        visual_motion_low=dc["visual_motion_low"],
        visual_motion_high=dc["visual_motion_high"],
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, collate_fn=_multimodal_collate
    )

    model = LateFusionBiGRUClassifier(
        visual_d_in=mc["visual_d_in"],
        audio_d_in=mc["audio_d_in"],
        hidden=mc["hidden"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
        pooling=mc["pooling"],
        fusion_hidden=mc["fusion_hidden"],
        use_visual=True,
        use_audio=True,
        use_text=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for visual_x, visual_len, audio_x, audio_len, y in test_loader:
            visual_x, visual_len = visual_x.to(device), visual_len.to(device)
            audio_x, audio_len = audio_x.to(device), audio_len.to(device)
            y = y.to(device)
            logits = model(visual_x, visual_len, audio_x, audio_len)
            total_loss += criterion(logits, y).item() * y.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    _print_metrics(total, total_loss / total, total_correct, all_preds, all_labels)


def evaluate_avt(
    ckpt: dict,
    root: str | None,
    opensmile_root: str | None,
    whisper_root: str | None,
    device: torch.device,
) -> None:
    mc = ckpt["model_config"]
    dc = ckpt["dataset_config"]

    # Infer active modalities from stored d_in values (None → disabled)
    use_visual = mc.get("visual_d_in") is not None
    use_audio = mc.get("audio_d_in") is not None
    use_text = mc.get("text_d_in") is not None

    openface_root = (root or dc["openface_root"]) if use_visual else None
    audio_root = (opensmile_root or dc["opensmile_root"]) if use_audio else None
    text_root = (whisper_root or dc["whisper_root"]) if use_text else None

    test_ds = MultimodalDatasetAVT(
        openface_root,
        audio_root,
        text_root,
        split="Test",
        visual_key_fn=openface_ur_lying_key,
        audio_key_fn=opensmile_ur_lying_key,
        audio_subsample_k=dc["audio_subsample_k"],
        visual_subsample_k=dc["visual_subsample_k"],
        audio_motion_method=dc["audio_motion_method"],
        audio_motion_low=dc["audio_motion_low"],
        audio_motion_high=dc["audio_motion_high"],
        visual_motion_method=dc["visual_motion_method"],
        visual_motion_low=dc["visual_motion_low"],
        visual_motion_high=dc["visual_motion_high"],
    )
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, collate_fn=_avt_collate
    )

    model = LateFusionBiGRUClassifier(
        visual_d_in=mc["visual_d_in"],
        audio_d_in=mc["audio_d_in"],
        text_d_in=mc["text_d_in"],
        hidden=mc["hidden"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
        pooling=mc["pooling"],
        fusion_hidden=mc["fusion_hidden"],
        use_visual=use_visual,
        use_audio=use_audio,
        use_text=use_text,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for (
            visual_x,
            visual_len,
            audio_x,
            audio_len,
            text_x,
            text_len,
            y,
        ) in test_loader:
            if visual_x is not None:
                visual_x, visual_len = visual_x.to(device), visual_len.to(device)
            if audio_x is not None:
                audio_x, audio_len = audio_x.to(device), audio_len.to(device)
            if text_x is not None:
                text_x, text_len = text_x.to(device), text_len.to(device)
            y = y.to(device)
            logits = model(
                visual_x=visual_x,
                visual_lengths=visual_len,
                audio_x=audio_x,
                audio_lengths=audio_len,
                text_x=text_x,
                text_lengths=text_len,
            )
            total_loss += criterion(logits, y).item() * y.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    _print_metrics(total, total_loss / total, total_correct, all_preds, all_labels)


def evaluate_pretrained_avt(
    ckpt: dict,
    clips_root: str | None,
    audio_raw_root: str | None,
    whisper_root: str | None,
    device: torch.device,
) -> None:
    from dataset.pretrained_multimodal_dataset import (
        PretrainedMultimodalDataset,
        _pretrained_collate,
    )
    from model.pretrained_encoders import LoRAEncoderConfig
    from model.pretrained_late_fusion import PretrainedLateFusionClassifier

    mc = ckpt["model_config"]
    dc = ckpt["dataset_config"]

    _clips_root = (clips_root or dc.get("clips_root")) if mc["use_visual"] else None
    _audio_root = (
        (audio_raw_root or dc.get("audio_raw_root")) if mc["use_audio"] else None
    )
    _text_root = (whisper_root or dc.get("whisper_root")) if mc["use_text"] else None

    test_ds = PretrainedMultimodalDataset(
        clips_root=_clips_root,
        audio_root=_audio_root,
        whisper_root=_text_root,
        split="Test",
        num_frames=dc.get("num_video_frames", 16),
        roberta_model_name=mc.get("text_model_name", "roberta-base"),
        roberta_max_length=dc.get("roberta_max_length", 512),
        max_audio_seconds=dc.get("max_audio_seconds", 60.0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=4,
        shuffle=False,
        collate_fn=_pretrained_collate,
        num_workers=0,
    )

    lora_cfg = LoRAEncoderConfig(
        r=mc["lora_r"],
        lora_alpha=mc["lora_alpha"],
        lora_dropout=mc["lora_dropout"],
    )
    model = PretrainedLateFusionClassifier(
        use_visual=mc["use_visual"],
        use_audio=mc["use_audio"],
        use_text=mc["use_text"],
        video_model_name=mc.get("video_model_name", "MCG-NJU/videomae-base"),
        audio_model_name=mc.get("audio_model_name", "facebook/wav2vec2-base-960h"),
        text_model_name=mc.get("text_model_name", "roberta-base"),
        lora_cfg=lora_cfg,
        fusion_hidden=mc.get("fusion_hidden", 256),
        dropout=mc.get("dropout", 0.3),
        text_pool=mc.get("text_pool", "cls"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    all_preds: list[float] = []
    all_labels: list[float] = []

    with torch.no_grad():
        for px, wf, wf_attn, ids, txt_attn, vl, y in test_loader:
            if px is not None:
                px = px.to(device)
            if wf is not None:
                wf = wf.to(device)
            if wf_attn is not None:
                wf_attn = wf_attn.to(device)
            if ids is not None:
                ids = ids.to(device)
            if txt_attn is not None:
                txt_attn = txt_attn.to(device)
            if vl is not None:
                vl = vl.to(device)
            y = y.to(device)
            logits = model(
                pixel_values=px,
                waveforms=wf,
                audio_attention_mask=wf_attn,
                input_ids=ids,
                text_attention_mask=txt_attn,
                video_lengths=vl,
            )
            total_loss += criterion(logits, y).item() * y.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            total_correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    _print_metrics(total, total_loss / total, total_correct, all_preds, all_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="best_bigru.pt", help="Path to checkpoint")
    parser.add_argument(
        "--root", default=None, help="Override OpenFace root (default: from checkpoint)"
    )
    parser.add_argument(
        "--opensmile_root",
        default=None,
        help="Override OpenSMILE root for multimodal checkpoints (default: from checkpoint)",
    )
    parser.add_argument(
        "--whisper_root",
        default=None,
        help="Override Whisper root for AVT checkpoints (default: from checkpoint)",
    )
    parser.add_argument(
        "--clips_root",
        default=None,
        help="Override clips_raw root for pretrained_avt checkpoints (default: from checkpoint)",
    )
    parser.add_argument(
        "--audio_raw_root",
        default=None,
        help="Override audio_raw root for pretrained_avt checkpoints (default: from checkpoint)",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    model_type = ckpt.get("model_type", "bilstm")

    print(f"Loaded   : {args.ckpt}")
    print(f"Type     : {model_type}")
    if "best_val_acc" in ckpt:
        print(f"Val acc  : {ckpt['best_val_acc']:.4f}  (epoch {ckpt['epoch']})")
        print(f"Val loss : {ckpt['best_val_loss']:.4f}")
    elif "val_acc" in ckpt:
        print(f"Val acc  : {ckpt['val_acc']:.4f}  (epoch {ckpt['epoch']})")
    elif "mean_val_acc" in ckpt:
        print(f"CV acc   : {ckpt['mean_val_acc']:.4f}  (avg epoch {ckpt['avg_epoch']})")
    print(f"Device   : {device}")

    if model_type == "pretrained_avt":
        evaluate_pretrained_avt(
            ckpt, args.clips_root, args.audio_raw_root, args.whisper_root, device
        )
    elif model_type == "multimodal_avt":
        evaluate_avt(ckpt, args.root, args.opensmile_root, args.whisper_root, device)
    elif model_type == "multimodal":
        evaluate_multimodal(ckpt, args.root, args.opensmile_root, device)
    else:
        evaluate_bilstm(ckpt, args.root, device)
