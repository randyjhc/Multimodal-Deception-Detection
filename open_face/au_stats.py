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

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

# Use non-interactive backend for saving plots
matplotlib.use("Agg")

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
        au_r_values: dict[str, list[float]] = {au: [] for au in au_r_cols}
        au_c_values: dict[str, list[float]] = {au: [] for au in au_c_cols}

        for row in reader:
            # Skip rows with insufficient data
            if (
                len(row)
                < max(
                    max(au_r_cols.values(), default=0),
                    max(au_c_cols.values(), default=0),
                )
                + 1
            ):
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


def plot_video_au_activation(
    video_stats: dict, video_name: str, metric_type: str, output_dir: Path
) -> None:
    """Plot per-video AU activation as horizontal bar chart.

    Args:
        video_stats: Per-video statistics dict from compute_video_au_stats().
        video_name: Display name for the plot title.
        metric_type: Either "regression" (intensity) or "classification" (frequency).
        output_dir: Directory to save the plot.
    """
    # Filter metrics by type
    if metric_type == "regression":
        suffix = "_r_intensity"
        ylabel = "AU Intensity (0-5 scale)"
    elif metric_type == "classification":
        suffix = "_c_frequency"
        ylabel = "AU Frequency (proportion of frames)"
    else:
        raise ValueError(f"Invalid metric_type: {metric_type}")

    # Extract AU names and values
    metrics = [(k, v) for k, v in video_stats.items() if k.endswith(suffix)]
    metrics.sort(key=lambda x: x[1], reverse=True)  # Sort descending by value

    if not metrics:
        print(f"Warning: No {metric_type} metrics found for {video_name}")
        return

    au_names = [m[0].replace(suffix, "") for m in metrics]
    values = [m[1] for m in metrics]

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    y_positions = range(len(au_names))

    ax.barh(y_positions, values, color="#1f77b4", alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(au_names, fontsize=10)
    ax.set_xlabel(ylabel, fontsize=12)
    ax.set_ylabel("Action Unit", fontsize=12)
    ax.set_title(
        f"AU {metric_type.capitalize()} - {video_name}", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"video_{video_name}_{metric_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_class_comparison(
    deceptive_summary: dict,
    truthful_summary: dict,
    deceptive_stats: list[dict],
    truthful_stats: list[dict],
    metric_type: str,
    output_dir: Path,
) -> None:
    """Plot per-class AU comparison as grouped bar chart with significance markers.

    Args:
        deceptive_summary: Class average dict from compute_class_average().
        truthful_summary: Class average dict from compute_class_average().
        deceptive_stats: List of per-video stats for deceptive class (for t-test).
        truthful_stats: List of per-video stats for truthful class (for t-test).
        metric_type: Either "regression" (intensity) or "classification" (frequency).
        output_dir: Directory to save the plot.
    """
    # Filter metrics by type
    if metric_type == "regression":
        suffix = "_r_intensity"
        ylabel = "Average AU Intensity (0-5 scale)"
    elif metric_type == "classification":
        suffix = "_c_frequency"
        ylabel = "Average AU Frequency (proportion of frames)"
    else:
        raise ValueError(f"Invalid metric_type: {metric_type}")

    # Extract AU names and values
    metrics = sorted([k for k in deceptive_summary.keys() if k.endswith(suffix)])

    if not metrics:
        print(f"Warning: No {metric_type} metrics found for class comparison")
        return

    au_names = [m.replace(suffix, "") for m in metrics]
    d_means = [deceptive_summary[m]["mean"] for m in metrics]
    d_stds = [deceptive_summary[m]["std"] for m in metrics]
    t_means = [truthful_summary[m]["mean"] for m in metrics]
    t_stds = [truthful_summary[m]["std"] for m in metrics]

    # Perform t-tests to identify significant differences
    p_values = []
    for metric in metrics:
        d_vals = [v[metric] for v in deceptive_stats]
        t_vals = [v[metric] for v in truthful_stats]
        _, p = stats.ttest_ind(d_vals, t_vals, equal_var=False)
        p_values.append(p)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 8))
    x = range(len(au_names))
    width = 0.35

    ax.bar(
        [i - width / 2 for i in x],
        d_means,
        width,
        yerr=d_stds,
        label="Deceptive",
        color="#d62728",
        alpha=0.7,
        capsize=3,
    )
    ax.bar(
        [i + width / 2 for i in x],
        t_means,
        width,
        yerr=t_stds,
        label="Truthful",
        color="#1f77b4",
        alpha=0.7,
        capsize=3,
    )

    # Add asterisk to x-labels for significant AUs (p < 0.05)
    au_labels = [
        f"{au_names[i]}*" if p_values[i] < 0.05 else au_names[i]
        for i in range(len(au_names))
    ]

    ax.set_xlabel("Action Unit", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"AU {metric_type.capitalize()} - Class Comparison (Deceptive vs Truthful)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(au_labels, fontsize=10, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"class_comparison_{metric_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_dataset_distribution(
    deceptive_stats: list[dict],
    truthful_stats: list[dict],
    metric_type: str,
    output_dir: Path,
) -> None:
    """Plot dataset-wide AU distribution as violin plot (combined classes).

    Args:
        deceptive_stats: List of per-video statistics for deceptive class.
        truthful_stats: List of per-video statistics for truthful class.
        metric_type: Either "regression" (intensity) or "classification" (frequency).
        output_dir: Directory to save the plot.
    """
    # Filter metrics by type
    if metric_type == "regression":
        suffix = "_r_intensity"
        ylabel = "AU Intensity (0-5 scale)"
    elif metric_type == "classification":
        suffix = "_c_frequency"
        ylabel = "AU Frequency (proportion of frames)"
    else:
        raise ValueError(f"Invalid metric_type: {metric_type}")

    # Extract AU names and collect data
    if not deceptive_stats or not truthful_stats:
        print(f"Warning: No data for {metric_type} distribution plot")
        return

    metrics = sorted([k for k in deceptive_stats[0].keys() if k.endswith(suffix)])

    if not metrics:
        print(f"Warning: No {metric_type} metrics found for distribution plot")
        return

    au_names = [m.replace(suffix, "") for m in metrics]

    # Prepare combined data for violin plot (merge both classes)
    data_to_plot = []
    positions = []

    for i, metric in enumerate(metrics):
        d_vals = [v[metric] for v in deceptive_stats]
        t_vals = [v[metric] for v in truthful_stats]
        # Combine both classes into single distribution
        combined_vals = d_vals + t_vals

        data_to_plot.append(combined_vals)
        positions.append(i * 1.5)

    # Create violin plot
    fig, ax = plt.subplots(figsize=(16, 8))

    parts = ax.violinplot(
        data_to_plot,
        positions=positions,
        widths=1.0,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Color the violin plots with a single neutral color
    for pc in parts["bodies"]:
        pc.set_facecolor("#2ca02c")  # Green color for combined distribution
        pc.set_alpha(0.5)

    # Overlay boxplot on top of violin plot
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=0.4,
        patch_artist=True,
        showmeans=False,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "alpha": 0.5,
            "markeredgecolor": "none",
            "markerfacecolor": "#2ca02c",
        },
    )

    # Color boxes to match the violin plots
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("#2ca02c")
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)

    # Color whiskers and caps
    for whisker in bp["whiskers"]:
        whisker.set_color("#2ca02c")
    for cap in bp["caps"]:
        cap.set_color("#2ca02c")

    # Set x-axis labels to AU names
    ax.set_xticks(positions)
    ax.set_xticklabels(au_names, fontsize=10, rotation=45, ha="right")
    ax.set_xlabel("Action Unit", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"AU {metric_type.capitalize()} - Combined Dataset Distribution",
        fontsize=14,
        fontweight="bold",
    )

    # Add legend
    legend_elements = [
        Patch(facecolor="#2ca02c", alpha=0.5, label="Combined (Deceptive + Truthful)"),
        Patch(facecolor="white", edgecolor="#2ca02c", linewidth=1.5, alpha=0.7, label="Boxplot"),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"dataset_distribution_{metric_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main(limit: int | None = None, visualize: bool = False):
    """Compute and display AU statistics with optional file limit and visualization.

    Args:
        limit: If specified, process only the first N files per class.
               If None, process all files.
        visualize: If True, generate visualization plots.
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

    print(
        f"\nProcessing {len(deceptive_files)} deceptive and {len(truthful_files)} truthful videos..."
    )

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
    print(
        f"Deceptive videos: {len(deceptive_files)}  |  Truthful videos: {len(truthful_files)}"
    )

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
    print(
        "AU Classification (Frequency): Proportion of frames with AU present (0-1 scale)"
    )
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

    # Generate visualizations if requested
    if visualize:
        print("\n" + "=" * 100)
        print("Generating Visualizations")
        print("=" * 100)

        # Create output directory for plots
        output_dir = Path("open_face/plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Per-Video Average Activation (first video from each class as examples)
        print("\n1. Per-Video AU Activation (Example Videos):")
        if deceptive_files and deceptive_stats:
            video_name = deceptive_files[0].stem
            plot_video_au_activation(
                deceptive_stats[0], f"{video_name}_deceptive", "regression", output_dir
            )
            plot_video_au_activation(
                deceptive_stats[0],
                f"{video_name}_deceptive",
                "classification",
                output_dir,
            )

        if truthful_files and truthful_stats:
            video_name = truthful_files[0].stem
            plot_video_au_activation(
                truthful_stats[0], f"{video_name}_truthful", "regression", output_dir
            )
            plot_video_au_activation(
                truthful_stats[0],
                f"{video_name}_truthful",
                "classification",
                output_dir,
            )

        # 2. Per-Class Average Activation (Comparison)
        print("\n2. Per-Class AU Comparison (Deceptive vs Truthful):")
        plot_class_comparison(
            d_summary,
            t_summary,
            deceptive_stats,
            truthful_stats,
            "regression",
            output_dir,
        )
        plot_class_comparison(
            d_summary,
            t_summary,
            deceptive_stats,
            truthful_stats,
            "classification",
            output_dir,
        )

        # 3. Whole Dataset Distribution
        print("\n3. Dataset-wide AU Distribution:")
        plot_dataset_distribution(
            deceptive_stats, truthful_stats, "regression", output_dir
        )
        plot_dataset_distribution(
            deceptive_stats, truthful_stats, "classification", output_dir
        )

        print(f"\n✓ All visualizations saved to: {output_dir}/")
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
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots (saved to open_face/plots/)",
    )

    args = parser.parse_args()
    main(limit=args.limit, visualize=args.visualize)
