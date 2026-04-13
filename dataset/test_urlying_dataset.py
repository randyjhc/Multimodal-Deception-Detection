#!/usr/bin/env python3
"""Smoke-test MultimodalDataset and make_multimodal_loaders() with the UR_LYING dataset."""

import argparse

import torch

from multimodal_dataset import (
    make_multimodal_loaders,
    openface_ur_lying_key,
    opensmile_ur_lying_key,
)

DEFAULT_OPENFACE_ROOT = "dataset/UR_LYING_Deception_Dataset/openface"
DEFAULT_OPENSMILE_ROOT = "dataset/UR_LYING_Deception_Dataset/opensmile"


def test_loader(name: str, loader: torch.utils.data.DataLoader) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openface-root",
        default=DEFAULT_OPENFACE_ROOT,
        help="Path to OpenFace features root",
    )
    parser.add_argument(
        "--opensmile-root",
        default=DEFAULT_OPENSMILE_ROOT,
        help="Path to OpenSMILE features root",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--visual-subsample-k",
        type=int,
        default=1,
        help="Visual subsampling factor (keep every k-th frame)",
    )
    parser.add_argument(
        "--audio-subsample-k",
        type=int,
        default=1,
        help="Audio subsampling factor (keep every k-th frame); default matches run_training_gru.py",
    )
    args = parser.parse_args()

    print(f"OpenFace root:  {args.openface_root}")
    print(f"OpenSMILE root: {args.opensmile_root}")
    print(
        f"visual_subsample_k={args.visual_subsample_k}  audio_subsample_k={args.audio_subsample_k}\n"
    )

    # --- loaders without subsampling (to measure "before" lengths) ---
    train_b, val_b, test_b, visual_d_in, audio_d_in = make_multimodal_loaders(
        openface_root=args.openface_root,
        opensmile_root=args.opensmile_root,
        batch_size=args.batch_size,
        visual_key_fn=openface_ur_lying_key,
        audio_key_fn=opensmile_ur_lying_key,
        visual_subsample_k=1,
        audio_subsample_k=1,
    )

    # --- loaders with requested subsampling (to measure "after" lengths) ---
    train, val, test, _, _ = make_multimodal_loaders(
        openface_root=args.openface_root,
        opensmile_root=args.opensmile_root,
        batch_size=args.batch_size,
        visual_key_fn=openface_ur_lying_key,
        audio_key_fn=opensmile_ur_lying_key,
        visual_subsample_k=args.visual_subsample_k,
        audio_subsample_k=args.audio_subsample_k,
        visual_motion_method="feature_diff",
        visual_motion_low=0.2,
        audio_motion_method="feature_diff",
        audio_motion_low=0.2,
    )

    print(
        f"visual_d_in={visual_d_in}  audio_d_in={audio_d_in} | "
        f"train={len(train)} batches  val={len(val)} batches  test={len(test)} batches\n"
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
    print()

    print("--- Loader smoke test (after subsampling) ---")
    test_loader("train", train)
    test_loader("val  ", val)
    test_loader("test ", test)
    print("OK")


if __name__ == "__main__":
    main()
