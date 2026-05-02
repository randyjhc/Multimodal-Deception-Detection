"""
Compute Action Unit (AU) statistics from OpenFace features.

Provides three levels of statistics:
  - Per-video average AU activation
  - Per-class average AU activation (Deceptive vs Truthful)
  - Dataset-level average activation (combined)
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

# ==== Constants ====

# Paths
OPENFACE_DIR = Path("dataset/UR_LYING_Deception_Dataset/openface_raw")
PLOTS_DIR = Path("plots")
# AU Metric Suffixes
AU_REGRESSION_SUFFIX = "_r"
AU_CLASSIFICATION_SUFFIX = "_c"
AU_INTENSITY_METRIC = "_r_intensity"
AU_FREQUENCY_METRIC = "_c_frequency"
# Metric Type Labels
REGRESSION_YLABEL = "AU Intensity (0-5 scale)"
CLASSIFICATION_YLABEL = "AU Frequency (proportion of frames)"
# Visualization
FIGURE_SIZE_SMALL = (12, 8)
FIGURE_SIZE_LARGE = (16, 8)
PLOT_DPI = 300
COLOR_DECEPTIVE = "#d62728"
COLOR_TRUTHFUL = "#1f77b4"
COLOR_COMBINED = "#2ca02c"
FONT_SIZE_SMALL = 10
FONT_SIZE_MEDIUM = 12
FONT_SIZE_LARGE = 14
BAR_WIDTH = 0.35
VIOLIN_WIDTH = 1.0
BOX_WIDTH = 0.4
VIOLIN_SPACING = 1.5
GRID_ALPHA = 0.3
METRIC_TYPES = ["regression", "classification"]


# ==== Utility Functions ====


def get_metric_config(metric_type):
    """
    Get suffix and ylabel for a given metric type.
    Returns: (suffix, ylabel).
    """
    if metric_type == "regression":
        return AU_INTENSITY_METRIC, REGRESSION_YLABEL
    elif metric_type == "classification":
        return AU_FREQUENCY_METRIC, CLASSIFICATION_YLABEL
    else:
        raise ValueError(f"Invalid metric_type: {metric_type}")


def extract_metric_values(stats_list, metric):
    return [stats[metric] for stats in stats_list]


def compute_ttest(deceptive_stats, truthful_stats, metric):
    d_vals = extract_metric_values(deceptive_stats, metric)
    t_vals = extract_metric_values(truthful_stats, metric)
    return stats.ttest_ind(d_vals, t_vals, equal_var=False)


def get_metrics_by_suffix(stats_dict, suffix):
    return sorted([k for k in stats_dict.keys() if k.endswith(suffix)])


def extract_au_name(metric):
    for suffix in [AU_INTENSITY_METRIC, AU_FREQUENCY_METRIC]:
        if metric.endswith(suffix):
            return metric.replace(suffix, "")
    return metric


def extract_summary_stats(summary, metrics):
    means = [summary[m]["mean"] for m in metrics]
    stds = [summary[m]["std"] for m in metrics]
    return means, stds


# ==== Data Processing Functions ====


def get_au_columns(header):
    au_r_cols = {}
    au_c_cols = {}

    for idx, col_name in enumerate(header):
        if col_name.startswith("AU") and col_name.endswith(AU_REGRESSION_SUFFIX):
            au_name = col_name[: -len(AU_REGRESSION_SUFFIX)]
            au_r_cols[au_name] = idx
        elif col_name.startswith("AU") and col_name.endswith(AU_CLASSIFICATION_SUFFIX):
            au_name = col_name[: -len(AU_CLASSIFICATION_SUFFIX)]
            au_c_cols[au_name] = idx

    return au_r_cols, au_c_cols


def compute_video_au_stats(filepath: Path) -> dict[str, float]:
    """
    Compute per-video AU statistics.
    For AU##_r: Computes average intensity across all frames.
    For AU##_c: Computes frequency of occurrence (proportion of frames where AU=1).
    Returns: Dict mapping AU metric names to their values.
    """
    with filepath.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        au_r_cols, au_c_cols = get_au_columns(header)

        # Collect values for each AU
        au_r_values: dict[str, list[float]] = {au: [] for au in au_r_cols}
        au_c_values: dict[str, list[float]] = {au: [] for au in au_c_cols}

        min_cols = (
            max(
                max(au_r_cols.values(), default=0),
                max(au_c_cols.values(), default=0),
            )
            + 1
        )

        for row in reader:
            # Skip rows with insufficient data
            if len(row) < min_cols:
                continue

            # Collect regression AU values
            for au, col_idx in au_r_cols.items():
                try:
                    value = float(row[col_idx])
                    au_r_values[au].append(value)
                except (ValueError, IndexError):
                    pass

            # Collect classification AU values
            for au, col_idx in au_c_cols.items():
                try:
                    value = float(row[col_idx])
                    au_c_values[au].append(value)
                except (ValueError, IndexError):
                    pass

    # Compute statistics
    stats_dict = {}

    for au_values, metric_suffix in [
        (au_r_values, AU_INTENSITY_METRIC),
        (au_c_values, AU_FREQUENCY_METRIC),
    ]:
        for au, values in au_values.items():
            stats_dict[f"{au}{metric_suffix}"] = (
                sum(values) / len(values) if values else 0.0
            )

    return stats_dict


def compute_class_average(video_stats_list):
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


def compute_dataset_average(deceptive_videos, truthful_videos):
    all_videos = deceptive_videos + truthful_videos

    if not all_videos:
        return {}

    metric_keys = all_videos[0].keys()
    dataset_avg = {}

    for metric in metric_keys:
        values = [video[metric] for video in all_videos]
        dataset_avg[metric] = sum(values) / len(values)

    return dataset_avg


# ==== Display Utilities ====


def print_au_statistics(
    d_summary,
    t_summary,
    deceptive_stats,
    truthful_stats,
    dataset_avg,
    suffix,
    section_title,
):
    """Print formatted AU statistics table for a given metric type."""
    print("\n" + "=" * 100)
    print(section_title)
    print("=" * 100)
    print(
        f"{'AU':<8} {'D Mean':>10}  {'D Std':>10}  {'T Mean':>10}  {'T Std':>10}  {'p-value':>12}  {'Dataset Avg':>12}"
    )
    print("-" * 100)

    metrics = get_metrics_by_suffix(d_summary, suffix)

    for metric in metrics:
        au_name = extract_au_name(metric)

        dm = d_summary[metric]["mean"]
        ds = d_summary[metric]["std"]
        tm = t_summary[metric]["mean"]
        ts = t_summary[metric]["std"]

        _, p = compute_ttest(deceptive_stats, truthful_stats, metric)
        dataset_mean = dataset_avg[metric]

        print(
            f"{au_name:<8} {dm:>10.4f}  {ds:>10.4f}  {tm:>10.4f}  {ts:>10.4f}  {p:>12.6f}  {dataset_mean:>12.4f}"
        )


# ==== Matplotlib Utilities ====


def save_and_close_plot(fig, output_path):
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def set_plot_labels(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_MEDIUM)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_MEDIUM)
    ax.set_title(title, fontsize=FONT_SIZE_LARGE, fontweight="bold")


# ==== Visualization Functions ====


def plot_video_au_activation(video_stats, video_name, metric_type, output_dir):
    """Plot per-video AU activation as horizontal bar chart."""
    suffix, ylabel = get_metric_config(metric_type)

    # Extract AU names and values
    metrics = [(k, v) for k, v in video_stats.items() if k.endswith(suffix)]
    metrics.sort(key=lambda x: x[1], reverse=True)  # Sort descending by value

    if not metrics:
        print(f"Warning: No {metric_type} metrics found for {video_name}")
        return

    au_names = [extract_au_name(m[0]) for m in metrics]
    values = [m[1] for m in metrics]

    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)
    y_positions = range(len(au_names))

    ax.barh(y_positions, values, color=COLOR_TRUTHFUL, alpha=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(au_names, fontsize=FONT_SIZE_SMALL)
    set_plot_labels(
        ax, ylabel, "Action Unit", f"AU {metric_type.capitalize()} - {video_name}"
    )
    ax.grid(axis="x", alpha=GRID_ALPHA)

    output_path = output_dir / f"video_{video_name}_{metric_type}.png"
    save_and_close_plot(fig, output_path)


def plot_class_comparison(
    deceptive_summary,
    truthful_summary,
    deceptive_stats,
    truthful_stats,
    metric_type,
    output_dir,
):
    """Plot per-class AU comparison as grouped bar chart with significance markers."""
    suffix, base_ylabel = get_metric_config(metric_type)
    ylabel = f"Average {base_ylabel}"

    # Extract AU names and values
    metrics = get_metrics_by_suffix(deceptive_summary, suffix)

    if not metrics:
        print(f"Warning: No {metric_type} metrics found for class comparison")
        return

    au_names = [extract_au_name(m) for m in metrics]
    d_means, d_stds = extract_summary_stats(deceptive_summary, metrics)
    t_means, t_stds = extract_summary_stats(truthful_summary, metrics)

    # Perform t-tests to identify significant differences
    p_values = [compute_ttest(deceptive_stats, truthful_stats, m)[1] for m in metrics]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
    x = range(len(au_names))

    ax.bar(
        [i - BAR_WIDTH / 2 for i in x],
        d_means,
        BAR_WIDTH,
        yerr=d_stds,
        label="Deceptive",
        color=COLOR_DECEPTIVE,
        alpha=0.7,
        capsize=3,
    )
    ax.bar(
        [i + BAR_WIDTH / 2 for i in x],
        t_means,
        BAR_WIDTH,
        yerr=t_stds,
        label="Truthful",
        color=COLOR_TRUTHFUL,
        alpha=0.7,
        capsize=3,
    )

    # Add asterisk to x-labels for significant AUs (p < 0.05)
    au_labels = [
        f"{au_names[i]}*" if p_values[i] < 0.05 else au_names[i]
        for i in range(len(au_names))
    ]

    set_plot_labels(
        ax,
        "Action Unit",
        ylabel,
        f"AU {metric_type.capitalize()} - Class Comparison (Deceptive vs Truthful)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(au_labels, fontsize=FONT_SIZE_SMALL, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=GRID_ALPHA)

    output_path = output_dir / f"class_comparison_{metric_type}.png"
    save_and_close_plot(fig, output_path)


def plot_dataset_distribution(
    deceptive_stats,
    truthful_stats,
    metric_type,
    output_dir,
):
    """Plot dataset-wide AU distribution as violin plot (combined classes)."""
    suffix, ylabel = get_metric_config(metric_type)

    # Extract AU names and collect data
    if not deceptive_stats or not truthful_stats:
        print(f"Warning: No data for {metric_type} distribution plot")
        return

    metrics = get_metrics_by_suffix(deceptive_stats[0], suffix)

    if not metrics:
        print(f"Warning: No {metric_type} metrics found for distribution plot")
        return

    au_names = [extract_au_name(m) for m in metrics]

    # Prepare combined data for violin plot (merge both classes)
    data_to_plot = []
    positions = []

    for i, metric in enumerate(metrics):
        d_vals = extract_metric_values(deceptive_stats, metric)
        t_vals = extract_metric_values(truthful_stats, metric)
        # Combine both classes into single distribution
        combined_vals = d_vals + t_vals

        data_to_plot.append(combined_vals)
        positions.append(i * VIOLIN_SPACING)

    # Create violin plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)

    parts = ax.violinplot(
        data_to_plot,
        positions=positions,
        widths=VIOLIN_WIDTH,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # Color the violin plots with a single neutral color
    for pc in parts["bodies"]:
        pc.set_facecolor(COLOR_COMBINED)
        pc.set_alpha(0.5)

    # Overlay boxplot on top of violin plot
    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=BOX_WIDTH,
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
            "markerfacecolor": COLOR_COMBINED,
        },
    )

    # Color boxes to match the violin plots
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor(COLOR_COMBINED)
        patch.set_alpha(0.7)
        patch.set_linewidth(1.5)

    # Color whiskers and caps
    for whisker in bp["whiskers"]:
        whisker.set_color(COLOR_COMBINED)
    for cap in bp["caps"]:
        cap.set_color(COLOR_COMBINED)

    # Set x-axis labels to AU names
    ax.set_xticks(positions)
    ax.set_xticklabels(au_names, fontsize=FONT_SIZE_SMALL, rotation=45, ha="right")
    set_plot_labels(
        ax,
        "Action Unit",
        ylabel,
        f"AU {metric_type.capitalize()} - Combined Dataset Distribution",
    )

    # Add legend
    legend_elements = [
        Patch(
            facecolor=COLOR_COMBINED, alpha=0.5, label="Combined (Deceptive + Truthful)"
        ),
        Patch(
            facecolor="white",
            edgecolor=COLOR_COMBINED,
            linewidth=1.5,
            alpha=0.7,
            label="Boxplot",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    output_path = output_dir / f"dataset_distribution_{metric_type}.png"
    save_and_close_plot(fig, output_path)


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
    print_au_statistics(
        d_summary,
        t_summary,
        deceptive_stats,
        truthful_stats,
        dataset_avg,
        AU_INTENSITY_METRIC,
        "AU Regression (Intensity): Average activation strength (0-5 scale)",
    )

    # Section 2: Classification AUs (Frequency)
    print_au_statistics(
        d_summary,
        t_summary,
        deceptive_stats,
        truthful_stats,
        dataset_avg,
        AU_FREQUENCY_METRIC,
        "AU Classification (Frequency): Proportion of frames with AU present (0-1 scale)",
    )

    print()

    # Generate visualizations if requested
    if visualize:
        print("\n" + "=" * 100)
        print("Generating Visualizations")
        print("=" * 100)

        # Create output directory for plots
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Per-Video Average Activation (first video from each class as examples)
        print("\n1. Per-Video AU Activation (Example Videos):")
        for metric_type in METRIC_TYPES:
            if deceptive_files and deceptive_stats:
                video_name = deceptive_files[0].stem
                plot_video_au_activation(
                    deceptive_stats[0],
                    f"{video_name}_deceptive",
                    metric_type,
                    PLOTS_DIR,
                )

            if truthful_files and truthful_stats:
                video_name = truthful_files[0].stem
                plot_video_au_activation(
                    truthful_stats[0], f"{video_name}_truthful", metric_type, PLOTS_DIR
                )

        # 2. Per-Class Average Activation (Comparison)
        print("\n2. Per-Class AU Comparison (Deceptive vs Truthful):")
        for metric_type in METRIC_TYPES:
            plot_class_comparison(
                d_summary,
                t_summary,
                deceptive_stats,
                truthful_stats,
                metric_type,
                PLOTS_DIR,
            )

        # 3. Whole Dataset Distribution
        print("\n3. Dataset-wide AU Distribution:")
        for metric_type in METRIC_TYPES:
            plot_dataset_distribution(
                deceptive_stats, truthful_stats, metric_type, PLOTS_DIR
            )

        print(f"\n All visualizations saved to: {PLOTS_DIR}/")
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
