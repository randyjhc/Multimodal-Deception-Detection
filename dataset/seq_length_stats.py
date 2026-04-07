"""Print sequence length statistics for Train and Test splits.

Usage:
    uv run python dataset/seq_length_stats.py
    uv run python dataset/seq_length_stats.py --root open_face/OpenFace_features/OpenFace_features
"""
import argparse
from typing import List, Optional

from dataset.openface_dataset import OpenFaceDataset

# ROOT = "open_face/OpenFace_features/OpenFace_features"
ROOT = "dataset/UR_LYING_Deception_Dataset/splits"


def compute_sequence_length_stats(
    root_dir: str,
    split: str = "Train",
    feature_cols: Optional[List[str]] = None,
    min_confidence: float = 0.5,
) -> dict:
    """Compute sequence length statistics (mean, min, max) for a dataset split."""
    ds = OpenFaceDataset(root_dir, split, feature_cols, min_confidence)
    lengths = [ds[i][0].shape[0] for i in range(len(ds))]
    mean_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)
    print(f"[{split}] n={len(lengths)}  mean={mean_len:.1f}  min={min_len}  max={max_len}")
    return {"mean": mean_len, "min": min_len, "max": max_len, "lengths": lengths}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    args = parser.parse_args()

    compute_sequence_length_stats(args.root, split="Train")
    compute_sequence_length_stats(args.root, split="Test")
