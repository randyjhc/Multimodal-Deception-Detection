#!/usr/bin/env python3
"""Smoke-test OpenFaceDataset and make_loaders() with the UR_LYING organized dataset."""

import argparse
from pathlib import Path

import torch

from openface_dataset import make_loaders

DEFAULT_ROOT = "dataset/UR_LYING_Deception_Dataset/splits"


def test_loader(name: str, loader: torch.utils.data.DataLoader) -> None:
    total_d = total_t = 0
    min_len = float("inf")
    max_len = 0
    first_shape: tuple[int, ...] | None = None

    for x, lengths, y in loader:
        if first_shape is None:
            first_shape = tuple(x.shape)
        total_d += int(y.sum().item())
        total_t += int((y == 0).sum().item())
        min_len = min(min_len, int(lengths.min().item()))
        max_len = max(max_len, int(lengths.max().item()))

    print(
        f"  {name}: {len(loader)} batches | "
        f"first batch {first_shape} | "
        f"seq lengths min={min_len} max={max_len} | "
        f"labels D={total_d} T={total_t}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help="Path to OpenFace features root (default: UR_LYING organized)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading dataset from: {args.root}")
    train, val, test, d_in = make_loaders(args.root, batch_size=args.batch_size)
    print(f"d_in={d_in}  train={len(train)} batches  val={len(val)} batches  test={len(test)} batches")

    test_loader("train", train)
    test_loader("val  ", val)
    test_loader("test ", test)
    print("OK")


if __name__ == "__main__":
    main()
