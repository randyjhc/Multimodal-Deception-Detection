"""Print sequence length statistics for Train and Test splits.

Usage:
    python -m dataset.seq_length_stats
    python -m dataset.seq_length_stats --root open_face/OpenFace_features/OpenFace_features
    python -m dataset.seq_length_stats --k 5
    python -m dataset.seq_length_stats --k 5 --motion_method feature_diff --motion_low 0.01 --motion_high 2.0
"""

import argparse
from typing import List, Literal, Optional

import torch
from dataset.multimodal_dataset import (
    make_multimodal_loaders,
    openface_ur_lying_key,
    opensmile_ur_lying_key,
)
from dataset.openface_dataset import OpenFaceDataset

# ROOT = "open_face/OpenFace_features/OpenFace_features"
ROOT = "dataset/UR_LYING_Deception_Dataset/splits"


def compute_sequence_length_stats(
    root_dir: str,
    split: str = "Train",
    feature_cols: Optional[List[str]] = None,
    min_confidence: float = 0.5,
    subsample_k: int = 1,
    motion_method: Literal["none", "feature_diff"] = "none",
    motion_low: float = 0.0,
    motion_high: float = float("inf"),
) -> dict:
    """Compute sequence length statistics (mean, min, max) for a dataset split."""
    ds = OpenFaceDataset(
        root_dir,
        split,
        feature_cols,
        min_confidence,
        subsample_k,
        motion_method=motion_method,
        motion_low=motion_low,
        motion_high=motion_high,
    )
    lengths = [ds[i][0].shape[0] for i in range(len(ds))]
    mean_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    motion_info = (
        f"  motion={motion_method}[{motion_low},{motion_high}]"
        if motion_method != "none"
        else ""
    )
    print(
        f"[{split}] n={len(lengths)}  mean={mean_len:.1f}  min={min_len}  max={max_len}{motion_info}"
    )
    return {"mean": mean_len, "min": min_len, "max": max_len, "lengths": lengths}


def test_loader(name: str, loader: torch.utils.data.DataLoader) -> None:
    """Iterate a multimodal loader and print per-batch shape and label counts."""
    total_d = total_t = 0
    v_min = a_min = float("inf")
    v_max = a_max = 0
    first_v_shape: tuple[int, ...] | None = None
    first_a_shape: tuple[int, ...] | None = None

    for visual_x, visual_len, audio_x, audio_len, y in loader:
        if first_v_shape is None:
            first_v_shape = tuple(visual_x.shape)
            first_a_shape = tuple(audio_x.shape)
        total_d += int(y.sum().item())
        total_t += int((y == 0).sum().item())
        v_min = min(v_min, int(visual_len.min().item()))
        v_max = max(v_max, int(visual_len.max().item()))
        a_min = min(a_min, int(audio_len.min().item()))
        a_max = max(a_max, int(audio_len.max().item()))

    print(
        f"  {name}: {len(loader)} batches | "
        f"visual {first_v_shape} len=[{v_min},{v_max}] | "
        f"audio {first_a_shape} len=[{a_min},{a_max}] | "
        f"labels D={total_d} T={total_t}"
    )


def collect_lengths(
    loader: torch.utils.data.DataLoader,
) -> dict[str, list[int]]:
    """Collect per-sample sequence lengths by iterating the underlying dataset."""
    visual_lens: list[int] = []
    audio_lens: list[int] = []
    ds = loader.dataset
    for i in range(len(ds)):  # type: ignore[arg-type]
        v, a, _ = ds[i]  # type: ignore[index]
        visual_lens.append(v.shape[0])
        audio_lens.append(a.shape[0])
    return {"visual": visual_lens, "audio": audio_lens}


def print_length_stats(
    split: str,
    before: dict[str, list[int]],
    after: dict[str, list[int]],
    vk: int,
    ak: int,
) -> None:
    for modality, k in (("visual", vk), ("audio", ak)):
        b = before[modality]
        a = after[modality]
        b_mean = sum(b) / len(b)
        a_mean = sum(a) / len(a)
        print(
            f"  [{split}] {modality} (k={k}): "
            f"before mean={b_mean:.0f} min={min(b)} max={max(b)}  |  "
            f"after  mean={a_mean:.0f} min={min(a)} max={max(a)}"
        )


DEFAULT_OPENFACE_ROOT = "dataset/UR_LYING_Deception_Dataset/openface"
DEFAULT_OPENSMILE_ROOT = "dataset/UR_LYING_Deception_Dataset/opensmile"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", default=ROOT, help="OpenFace-only root (single-modality mode)"
    )
    parser.add_argument(
        "--openface-root",
        default=DEFAULT_OPENFACE_ROOT,
        help="OpenFace root (multimodal mode)",
    )
    parser.add_argument(
        "--opensmile-root",
        default=DEFAULT_OPENSMILE_ROOT,
        help="OpenSMILE root (multimodal mode)",
    )
    parser.add_argument(
        "--visual-subsample-k", type=int, default=1, help="Visual subsampling factor"
    )
    parser.add_argument(
        "--audio-subsample-k", type=int, default=1, help="Audio subsampling factor"
    )
    parser.add_argument(
        "--motion_method", default="none", choices=["none", "feature_diff"]
    )
    parser.add_argument("--motion_low", type=float, default=0.0)
    parser.add_argument("--motion_high", type=float, default=float("inf"))
    args = parser.parse_args()

    # --- loaders without subsampling (before) ---
    train_b, val_b, test_b, _, _ = make_multimodal_loaders(
        openface_root=args.openface_root,
        opensmile_root=args.opensmile_root,
        visual_key_fn=openface_ur_lying_key,
        audio_key_fn=opensmile_ur_lying_key,
        visual_subsample_k=1,
        audio_subsample_k=1,
    )

    # --- loaders with requested subsampling (after) ---
    train, val, test, _, _ = make_multimodal_loaders(
        openface_root=args.openface_root,
        opensmile_root=args.opensmile_root,
        visual_key_fn=openface_ur_lying_key,
        audio_key_fn=opensmile_ur_lying_key,
        visual_subsample_k=args.visual_subsample_k,
        audio_subsample_k=args.audio_subsample_k,
        visual_motion_method=args.motion_method,
        visual_motion_low=args.motion_low,
        visual_motion_high=args.motion_high,
        audio_motion_method=args.motion_method,
        audio_motion_low=args.motion_low,
        audio_motion_high=args.motion_high,
    )

    print(
        f"visual_subsample_k={args.visual_subsample_k}  "
        f"audio_subsample_k={args.audio_subsample_k}  "
        f"motion={args.motion_method}[{args.motion_low},{args.motion_high}]\n"
    )
    print("--- Sequence length: before vs after subsampling ---")
    for name, lb, la in (
        ("train", train_b, train),
        ("val  ", val_b, val),
        ("test ", test_b, test),
    ):
        before = collect_lengths(lb)
        after = collect_lengths(la)
        print_length_stats(
            name, before, after, args.visual_subsample_k, args.audio_subsample_k
        )

    print("\n--- Loader smoke test ---")
    for name, loader in (("train", train), ("val  ", val), ("test ", test)):
        test_loader(name, loader)
    print("OK")
