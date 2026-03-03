"""
Compute Action Unit (AU) statistics from OpenFace features.

Extracts and analyzes AU activations for Deceptive vs. Truthful classes:
  - AU##_r (regression): Average intensity across frames (0-5 scale)
  - AU##_c (classification): Frequency of occurrence (proportion of frames with AU=1)

Provides three levels of statistics:
  A) Per-video average AU activation
  B) Per-class average AU activation (Deceptive vs Truthful)
  C) Dataset-level average activation (combined)
"""

import argparse
import csv
from pathlib import Path

from scipy import stats

OPENFACE_DIR = Path("open_face/OpenFace_features/OpenFace_features")


def get_au_columns(header: list[str]) -> tuple[dict[str, int], dict[str, int]]:
    """Extract AU column indices from CSV header.

    Returns:
        Tuple of (regression_aus, classification_aus) where each is a dict
        mapping AU name (e.g., 'AU01') to column index.
    """
    au_r_cols = {}
    au_c_cols = {}

    for idx, col_name in enumerate(header):
        if col_name.startswith("AU") and col_name.endswith("_r"):
            au_name = col_name[:-2]  # Remove '_r' suffix
            au_r_cols[au_name] = idx
        elif col_name.startswith("AU") and col_name.endswith("_c"):
            au_name = col_name[:-2]  # Remove '_c' suffix
            au_c_cols[au_name] = idx

    return au_r_cols, au_c_cols


def compute_video_au_stats(filepath: Path) -> dict:
    """Compute per-video AU statistics.

    For AU##_r: Computes average intensity across all frames.
    For AU##_c: Computes frequency of occurrence (proportion of frames where AU=1).

    Returns:
        Dict with keys for each AU (both _r and _c metrics).
    """
    with filepath.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        au_r_cols, au_c_cols = get_au_columns(header)

        # Collect values for each AU
        au_r_values = {au: [] for au in au_r_cols}
        au_c_values = {au: [] for au in au_c_cols}

        for row in reader:
            # Skip rows with insufficient data
            if len(row) < max(max(au_r_cols.values(), default=0), max(au_c_cols.values(), default=0)) + 1:
                continue

            # Collect regression AU values (intensity)
            for au, col_idx in au_r_cols.items():
                try:
                    value = float(row[col_idx])
                    au_r_values[au].append(value)
                except (ValueError, IndexError):
                    pass

            # Collect classification AU values (binary)
            for au, col_idx in au_c_cols.items():
                try:
                    value = float(row[col_idx])
                    au_c_values[au].append(value)
                except (ValueError, IndexError):
                    pass

    # Compute statistics
    stats_dict = {}

    # Regression AUs: average intensity
    for au, values in au_r_values.items():
        if values:
            stats_dict[f"{au}_r_intensity"] = sum(values) / len(values)
        else:
            stats_dict[f"{au}_r_intensity"] = 0.0

    # Classification AUs: frequency of occurrence
    for au, values in au_c_values.items():
        if values:
            stats_dict[f"{au}_c_frequency"] = sum(values) / len(values)
        else:
            stats_dict[f"{au}_c_frequency"] = 0.0

    return stats_dict


def compute_class_average(video_stats_list: list[dict]) -> dict:
    """Compute per-class average AU statistics.

    Args:
        video_stats_list: List of per-video statistics dicts.

    Returns:
        Dict mapping each AU metric to {'mean': float, 'std': float}.
    """
    if not video_stats_list:
        return {}

    # Get all metric keys from first video
    metric_keys = video_stats_list[0].keys()
    summary = {}

    for metric in metric_keys:
        values = [video[metric] for video in video_stats_list]
        mean_val = sum(values) / len(values)
        # Compute standard deviation
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        std_val = variance**0.5

        summary[metric] = {"mean": mean_val, "std": std_val}

    return summary


def compute_dataset_average(
    deceptive_videos: list[dict], truthful_videos: list[dict]
) -> dict:
    """Compute dataset-level average AU statistics.

    Args:
        deceptive_videos: List of per-video statistics for deceptive class.
        truthful_videos: List of per-video statistics for truthful class.

    Returns:
        Dict mapping each AU metric to overall mean across all videos.
    """
    all_videos = deceptive_videos + truthful_videos

    if not all_videos:
        return {}

    metric_keys = all_videos[0].keys()
    dataset_avg = {}

    for metric in metric_keys:
        values = [video[metric] for video in all_videos]
        dataset_avg[metric] = sum(values) / len(values)

    return dataset_avg


def main(limit: int | None = None):
    """Compute and display AU statistics with optional file limit.

    Args:
        limit: If specified, process only the first N files per class.
               If None, process all files.
    """
    # Collect file paths from Train and Test directories
    deceptive_files = []
    truthful_files = []

    for split in ["Train", "Test"]:
        deceptive_dir = OPENFACE_DIR / split / "Deceptive"
        truthful_dir = OPENFACE_DIR / split / "Truthful"

        if deceptive_dir.exists():
            deceptive_files.extend(sorted(deceptive_dir.glob("*.csv")))
        if truthful_dir.exists():
            truthful_files.extend(sorted(truthful_dir.glob("*.csv")))

    # Apply limit if specified
    if limit is not None:
        deceptive_files = deceptive_files[:limit]
        truthful_files = truthful_files[:limit]

    print(f"\nProcessing {len(deceptive_files)} deceptive and {len(truthful_files)} truthful videos...")

    # Compute per-video statistics
    print("Computing per-video statistics...")
    deceptive_stats = [compute_video_au_stats(f) for f in deceptive_files]
    truthful_stats = [compute_video_au_stats(f) for f in truthful_files]

    # Compute per-class averages
    print("Computing per-class averages...")
    d_summary = compute_class_average(deceptive_stats)
    t_summary = compute_class_average(truthful_stats)

    # Compute dataset-level averages
    print("Computing dataset-level averages...")
    dataset_avg = compute_dataset_average(deceptive_stats, truthful_stats)

    # Display results
    print("\n" + "=" * 100)
    print("Action Unit (AU) Statistics: Deceptive vs Truthful")
    print("=" * 100)
    print(f"Deceptive videos: {len(deceptive_files)}  |  Truthful videos: {len(truthful_files)}")

    # Section 1: Regression AUs (Intensity)
    print("\n" + "=" * 100)
    print("AU Regression (Intensity): Average activation strength (0-5 scale)")
    print("=" * 100)
    print(
        f"{'AU':<8} {'D Mean':>10}  {'D Std':>10}  {'T Mean':>10}  {'T Std':>10}  {'p-value':>12}  {'Dataset Avg':>12}"
    )
    print("-" * 100)

    # Get all regression AU metrics (those ending with _r_intensity)
    r_metrics = sorted([k for k in d_summary.keys() if k.endswith("_r_intensity")])

    for metric in r_metrics:
        au_name = metric.replace("_r_intensity", "")

        dm = d_summary[metric]["mean"]
        ds = d_summary[metric]["std"]
        tm = t_summary[metric]["mean"]
        ts = t_summary[metric]["std"]

        # Perform t-test
        d_vals = [v[metric] for v in deceptive_stats]
        t_vals = [v[metric] for v in truthful_stats]
        _, p = stats.ttest_ind(d_vals, t_vals, equal_var=False)

        dataset_mean = dataset_avg[metric]

        print(
            f"{au_name:<8} {dm:>10.4f}  {ds:>10.4f}  {tm:>10.4f}  {ts:>10.4f}  {p:>12.6f}  {dataset_mean:>12.4f}"
        )

    # Section 2: Classification AUs (Frequency)
    print("\n" + "=" * 100)
    print("AU Classification (Frequency): Proportion of frames with AU present (0-1 scale)")
    print("=" * 100)
    print(
        f"{'AU':<8} {'D Mean':>10}  {'D Std':>10}  {'T Mean':>10}  {'T Std':>10}  {'p-value':>12}  {'Dataset Avg':>12}"
    )
    print("-" * 100)

    # Get all classification AU metrics (those ending with _c_frequency)
    c_metrics = sorted([k for k in d_summary.keys() if k.endswith("_c_frequency")])

    for metric in c_metrics:
        au_name = metric.replace("_c_frequency", "")

        dm = d_summary[metric]["mean"]
        ds = d_summary[metric]["std"]
        tm = t_summary[metric]["mean"]
        ts = t_summary[metric]["std"]

        # Perform t-test
        d_vals = [v[metric] for v in deceptive_stats]
        t_vals = [v[metric] for v in truthful_stats]
        _, p = stats.ttest_ind(d_vals, t_vals, equal_var=False)

        dataset_mean = dataset_avg[metric]

        print(
            f"{au_name:<8} {dm:>10.4f}  {ds:>10.4f}  {tm:>10.4f}  {ts:>10.4f}  {p:>12.6f}  {dataset_mean:>12.4f}"
        )

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Action Unit statistics from OpenFace features"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process per class (for testing)",
    )

    args = parser.parse_args()
    main(limit=args.limit)
