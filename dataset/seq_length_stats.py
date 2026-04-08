"""Print sequence length statistics for Train and Test splits.

Usage:
    python -m dataset.seq_length_stats
    python -m dataset.seq_length_stats --root open_face/OpenFace_features/OpenFace_features
    python -m dataset.seq_length_stats --k 5
    python -m dataset.seq_length_stats --k 5 --motion_method feature_diff --motion_low 0.01 --motion_high 2.0
"""

import argparse
from typing import List, Literal, Optional

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Subsample every k-th frame (1 = no subsampling)",
    )
    parser.add_argument(
        "--motion_method", default="none", choices=["none", "feature_diff"]
    )
    parser.add_argument("--motion_low", type=float, default=0.0)
    parser.add_argument("--motion_high", type=float, default=float("inf"))
    args = parser.parse_args()

    compute_sequence_length_stats(
        args.root,
        split="Train",
        subsample_k=args.k,
        motion_method=args.motion_method,
        motion_low=args.motion_low,
        motion_high=args.motion_high,
    )
    compute_sequence_length_stats(
        args.root,
        split="Test",
        subsample_k=args.k,
        motion_method=args.motion_method,
        motion_low=args.motion_low,
        motion_high=args.motion_high,
    )
