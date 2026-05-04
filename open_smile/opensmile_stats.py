"""
Compute opensmile statistics from csv features.

Assumes a single aggregated csv (or multiple csvs with the same schema) where each row is fame-level
video/audio features and contains a clip identifier.

Default data structure:
    data/opensmile/

Read all csv files under this directory.

Features are grouped into interpretable categories such as:
    Pitch / F0
    Loudness / Energy
    Spectral / MFCC
    Voice quality
    Formants
    Voicing / speaking rate

Outputs:
  A) Per-class mean/std for each feature
  B) Dataset-level mean for each feature
  C) Welch's t-test between Deceptive vs Truthful per feature
  D) Visualization (Optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

OPENSMILE_DIR = Path("dataset/UR_LYING_Deception_Dataset/opensmile_raw")
PLOTS_DIR = OPENSMILE_DIR / "plots"
INTERNAL_CLIP_ID_COL = "__clip_id"
INTERNAL_SOURCE_PATH_COL = "__source_path"
GENERATED_OUTPUT_FILES = {
    "opensmile_feature_stats.csv",
    "opensmile_group_summary.csv",
}
ID_CANDIDATES = [
    INTERNAL_CLIP_ID_COL,
    INTERNAL_SOURCE_PATH_COL,
    "video_stem",
    "file",
    "filename",
    "clip_id",
    "id",
    "name",
]
LABEL_CANDIDATES = ["label", "class", "target", "y"]
POSITIVE_LABELS = {"deceptive", "lie", "lying", "1", 1}
NEGATIVE_LABELS = {"truthful", "truth", "0", 0}

FIGURE_SIZE_SMALL = (12, 8)
FIGURE_SIZE_LARGE = (16, 8)
PLOT_DPI = 300
FONT_SIZE_SMALL = 10
FONT_SIZE_MEDIUM = 12
FONT_SIZE_LARGE = 14
GRID_ALPHA = 0.3
BAR_WIDTH = 0.35
BOX_WIDTH = 0.4
VIOLIN_WIDTH = 0.8
TOP_K_DEFAULT = 20
MAX_DISTRIBUTION_POINTS = 5000
RANDOM_SEED = 42

COLOR_DECEPTIVE = "#d62728"
COLOR_TRUTHFUL = "#1f77b4"
COLOR_COMBINED = "#2ca02c"

# Constants

FEATURE_GROUP_PATTERNS = {
    "Pitch/F0": ["F0", "semitone", "logRelF0"],
    "Loudness/Energy": ["loudness", "equivalentSoundLevel"],
    "Spectral": ["spectralFlux", "alphaRatio", "hammarbergIndex", "slope"],
    "MFCC": ["mfcc"],
    "VoiceQuality": ["jitter", "shimmer", "HNR"],
    "Formants": ["F1", "F2", "F3", "bandwidth", "amplitudeLogRelF0"],
    "VoicingSegments": [
        "VoicedSegmentsPerSec",
        "MeanVoicedSegmentLength",
        "StddevVoicedSegmentLength",
        "MeanUnvoicedSegmentLength",
        "StddevUnvoicedSegmentLength",
        "loudnessPeaksPerSec",
    ],
}

# Helper Functions

def infer_group(feature_name: str) -> str:
    for group, patterns in FEATURE_GROUP_PATTERNS.items():
        if any(p in feature_name for p in patterns):
            return group
    return "Other"



def set_plot_labels(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=FONT_SIZE_MEDIUM)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZE_MEDIUM)
    ax.set_title(title, fontsize=FONT_SIZE_LARGE, fontweight="bold")



def save_and_close_plot(fig, output_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")



def choose_id_column(df: pd.DataFrame) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in ID_CANDIDATES:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None



def choose_label_column(df: pd.DataFrame) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for c in LABEL_CANDIDATES:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None



def normalize_label(value) -> str | None:
    if pd.isna(value):
        return None
    v = str(value).strip().lower()
    if v in {str(x).lower() for x in POSITIVE_LABELS}:
        return "Deceptive"
    if v in {str(x).lower() for x in NEGATIVE_LABELS}:
        return "Truthful"
    return None



def get_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    exclude = {INTERNAL_CLIP_ID_COL, INTERNAL_SOURCE_PATH_COL}
    id_col = choose_id_column(df)
    label_col = choose_label_column(df)
    if id_col:
        exclude.add(id_col)
    if label_col:
        exclude.add(label_col)

    numeric_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            numeric_cols.append(col)
    return numeric_cols



def infer_class_from_path(path: Path) -> str | None:
    """Infer class label from filename or any parent directory name."""
    parts = [part.lower() for part in path.parts]
    if any("deceptive" == part or "deceptive" in part for part in parts):
        return "Deceptive"
    if any("truthful" == part or "truthful" in part for part in parts):
        return "Truthful"
    return None



def load_csvs_from_directory(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    deceptive_frames = []
    truthful_frames = []

    csv_files = sorted(
        path for path in data_dir.rglob("*.csv") if path.name not in GENERATED_OUTPUT_FILES
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    for path in csv_files:
        df = pd.read_csv(path)
        df[INTERNAL_CLIP_ID_COL] = path.stem
        df[INTERNAL_SOURCE_PATH_COL] = str(path)

        inferred_class = infer_class_from_path(path)
        if inferred_class == "Deceptive":
            deceptive_frames.append(df)
            continue
        if inferred_class == "Truthful":
            truthful_frames.append(df)
            continue

        label_col = choose_label_column(df)
        if label_col is None:
            print(f"Skipping {path}: cannot infer class from path, filename, or label column")
            continue

        normalized = df[label_col].apply(normalize_label)
        if normalized.notna().sum() == 0:
            print(f"Skipping {path}: label column found but values were not recognized")
            continue

        deceptive_frames.append(df.loc[normalized == "Deceptive"].copy())
        truthful_frames.append(df.loc[normalized == "Truthful"].copy())

    if not deceptive_frames or not truthful_frames:
        raise ValueError(
            "Could not build both deceptive and truthful datasets. "
            "Use folder names / filenames containing 'deceptive' or 'truthful', or include a label column."
        )

    deceptive_df = pd.concat(deceptive_frames, ignore_index=True)
    truthful_df = pd.concat(truthful_frames, ignore_index=True)
    return deceptive_df, truthful_df



def count_clips(df: pd.DataFrame) -> int:
    """Count clips rather than frame-level openSMILE rows."""
    if INTERNAL_SOURCE_PATH_COL in df.columns:
        return int(df[INTERNAL_SOURCE_PATH_COL].nunique())

    id_col = choose_id_column(df)
    if id_col:
        return int(df[id_col].nunique())

    return len(df)



def limit_by_clips(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Keep all frame rows for the first N clips."""
    if limit <= 0:
        return df.iloc[0:0].copy()

    if INTERNAL_SOURCE_PATH_COL in df.columns:
        selected = df[INTERNAL_SOURCE_PATH_COL].drop_duplicates().head(limit)
        return df[df[INTERNAL_SOURCE_PATH_COL].isin(selected)].copy()

    id_col = choose_id_column(df)
    if id_col:
        selected = df[id_col].drop_duplicates().head(limit)
        return df[df[id_col].isin(selected)].copy()

    return df.head(limit).copy()



def prepare_feature_frames(
    deceptive_df: pd.DataFrame, truthful_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    common_cols = [c for c in deceptive_df.columns if c in truthful_df.columns]
    deceptive_df = deceptive_df[common_cols].copy()
    truthful_df = truthful_df[common_cols].copy()

    feature_cols = [c for c in get_numeric_feature_columns(deceptive_df) if c in common_cols]
    if not feature_cols:
        raise ValueError("No numeric openSMILE feature columns detected.")

    deceptive_features = deceptive_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    truthful_features = truthful_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    return deceptive_features, truthful_features, feature_cols



def compute_feature_statistics(
    deceptive_features: pd.DataFrame, truthful_features: pd.DataFrame, feature_cols: list[str]
) -> pd.DataFrame:
    rows = []

    for feature in feature_cols:
        d_vals = deceptive_features[feature].dropna()
        t_vals = truthful_features[feature].dropna()

        d_mean = d_vals.mean() if len(d_vals) else float("nan")
        d_std = d_vals.std(ddof=0) if len(d_vals) else float("nan")
        t_mean = t_vals.mean() if len(t_vals) else float("nan")
        t_std = t_vals.std(ddof=0) if len(t_vals) else float("nan")
        dataset_vals = pd.concat([d_vals, t_vals], ignore_index=True)
        dataset_mean = dataset_vals.mean() if len(dataset_vals) else float("nan")

        if len(d_vals) >= 2 and len(t_vals) >= 2:
            t_stat, p_value = stats.ttest_ind(d_vals, t_vals, equal_var=False, nan_policy="omit")
        else:
            t_stat, p_value = float("nan"), float("nan")

        diff = d_mean - t_mean
        abs_diff = abs(diff) if pd.notna(diff) else float("nan")

        rows.append(
            {
                "feature": feature,
                "group": infer_group(feature),
                "deceptive_mean": d_mean,
                "deceptive_std": d_std,
                "truthful_mean": t_mean,
                "truthful_std": t_std,
                "dataset_mean": dataset_mean,
                "mean_diff": diff,
                "abs_mean_diff": abs_diff,
                "t_stat": t_stat,
                "p_value": p_value,
                "n_deceptive": len(d_vals),
                "n_truthful": len(t_vals),
            }
        )

    result_df = pd.DataFrame(rows)
    return result_df.sort_values(["abs_mean_diff", "p_value"], ascending=[False, True])



def compute_group_summary(stats_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        stats_df.groupby("group", dropna=False)
        .agg(
            num_features=("feature", "count"),
            mean_abs_diff=("abs_mean_diff", "mean"),
            mean_p_value=("p_value", "mean"),
            significant_count=("p_value", lambda s: int((s < 0.05).sum())),
        )
        .reset_index()
        .sort_values("mean_abs_diff", ascending=False)
    )
    return summary



def print_feature_statistics(stats_df: pd.DataFrame, top_k: int = TOP_K_DEFAULT) -> None:
    print("\n" + "=" * 120)
    print(f"Top {min(top_k, len(stats_df))} openSMILE Features by Absolute Mean Difference")
    print("=" * 120)
    print(
        f"{'Feature':<45} {'Group':<18} {'D Mean':>10} {'D Std':>10} {'T Mean':>10} {'T Std':>10} {'Diff':>10} {'p-value':>12}"
    )
    print("-" * 120)

    for _, row in stats_df.head(top_k).iterrows():
        print(
            f"{row['feature']:<45.45} {row['group']:<18.18} "
            f"{row['deceptive_mean']:>10.4f} {row['deceptive_std']:>10.4f} "
            f"{row['truthful_mean']:>10.4f} {row['truthful_std']:>10.4f} "
            f"{row['mean_diff']:>10.4f} {row['p_value']:>12.6f}"
        )



def print_group_summary(group_df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("Feature Group Summary")
    print("=" * 90)
    print(
        f"{'Group':<20} {'#Features':>10} {'Mean |Diff|':>15} {'Mean p-value':>15} {'#p<0.05':>12}"
    )
    print("-" * 90)
    for _, row in group_df.iterrows():
        print(
            f"{row['group']:<20} {int(row['num_features']):>10d} "
            f"{row['mean_abs_diff']:>15.4f} {row['mean_p_value']:>15.6f} {int(row['significant_count']):>12d}"
        )



def plot_top_feature_comparison(stats_df: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    plot_df = stats_df.head(top_k).copy().iloc[::-1]
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_LARGE)
    y = range(len(plot_df))

    ax.barh(
        [i - BAR_WIDTH / 2 for i in y],
        plot_df["deceptive_mean"],
        BAR_WIDTH,
        xerr=plot_df["deceptive_std"],
        label="Deceptive",
        color=COLOR_DECEPTIVE,
        alpha=0.7,
        capsize=3,
    )
    ax.barh(
        [i + BAR_WIDTH / 2 for i in y],
        plot_df["truthful_mean"],
        BAR_WIDTH,
        xerr=plot_df["truthful_std"],
        label="Truthful",
        color=COLOR_TRUTHFUL,
        alpha=0.7,
        capsize=3,
    )

    labels = [
        f"{f}*" if pd.notna(p) and p < 0.05 else f
        for f, p in zip(plot_df["feature"], plot_df["p_value"])
    ]
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=FONT_SIZE_SMALL)
    set_plot_labels(ax, "Mean Feature Value", "Feature", f"Top {top_k} openSMILE Feature Comparison")
    ax.legend(fontsize=11)
    ax.grid(axis="x", alpha=GRID_ALPHA)

    output_path = output_dir / "opensmile_top_feature_comparison.png"
    save_and_close_plot(fig, output_path)



def plot_group_summary(group_df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)
    x = range(len(group_df))
    ax.bar(x, group_df["mean_abs_diff"], alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(group_df["group"], rotation=45, ha="right", fontsize=FONT_SIZE_SMALL)
    set_plot_labels(ax, "Feature Group", "Mean Absolute Difference", "Feature Group Separation Strength")
    ax.grid(axis="y", alpha=GRID_ALPHA)

    output_path = output_dir / "opensmile_group_summary.png"
    save_and_close_plot(fig, output_path)



def plot_feature_distribution(
    deceptive_features: pd.DataFrame,
    truthful_features: pd.DataFrame,
    stats_df: pd.DataFrame,
    output_dir: Path,
    top_k: int,
    max_points: int = MAX_DISTRIBUTION_POINTS,
) -> None:
    selected = stats_df.head(min(top_k, 10))["feature"].tolist()
    if not selected:
        return

    data_to_plot = []
    positions = []
    labels = []
    colors = []
    pos = 0

    def values_for_plot(series: pd.Series) -> list[float]:
        values = series.dropna()
        if max_points > 0 and len(values) > max_points:
            values = values.sample(n=max_points, random_state=RANDOM_SEED)
        return values.tolist()

    for feature in selected:
        d_vals = values_for_plot(deceptive_features[feature])
        t_vals = values_for_plot(truthful_features[feature])
        if d_vals:
            data_to_plot.append(d_vals)
            positions.append(pos)
            labels.append(f"{feature}\nD")
            colors.append(COLOR_DECEPTIVE)
            pos += 1
        if t_vals:
            data_to_plot.append(t_vals)
            positions.append(pos)
            labels.append(f"{feature}\nT")
            colors.append(COLOR_TRUTHFUL)
            pos += 1
        pos += 0.6

    fig, ax = plt.subplots(figsize=(18, 8))
    parts = ax.violinplot(
        data_to_plot,
        positions=positions,
        widths=VIOLIN_WIDTH,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.45)

    bp = ax.boxplot(
        data_to_plot,
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.4},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_alpha(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FONT_SIZE_SMALL)
    set_plot_labels(ax, "Feature / Class", "Feature Value Distribution", "openSMILE Distribution of Top Features")
    ax.legend(
        handles=[
            Patch(facecolor=COLOR_DECEPTIVE, alpha=0.45, label="Deceptive"),
            Patch(facecolor=COLOR_TRUTHFUL, alpha=0.45, label="Truthful"),
        ],
        fontsize=11,
        loc="upper right",
    )
    ax.grid(axis="y", alpha=GRID_ALPHA)

    output_path = output_dir / "opensmile_feature_distribution.png"
    save_and_close_plot(fig, output_path)



def main(
    limit: int | None = None,
    visualize: bool = False,
    top_k: int = TOP_K_DEFAULT,
    distribution_max_points: int = MAX_DISTRIBUTION_POINTS,
):
    deceptive_df, truthful_df = load_csvs_from_directory(OPENSMILE_DIR)

    if limit is not None:
        deceptive_df = limit_by_clips(deceptive_df, limit)
        truthful_df = limit_by_clips(truthful_df, limit)

    deceptive_features, truthful_features, feature_cols = prepare_feature_frames(
        deceptive_df, truthful_df
    )

    print("\n" + "=" * 100)
    print("openSMILE Statistics: Deceptive vs Truthful")
    print("=" * 100)
    print(
        f"Deceptive clips: {count_clips(deceptive_df)}  |  "
        f"Truthful clips: {count_clips(truthful_df)}"
    )
    print(f"Frame rows: Deceptive {len(deceptive_df)}  |  Truthful {len(truthful_df)}")
    print(f"Numeric features: {len(feature_cols)}")

    stats_df = compute_feature_statistics(deceptive_features, truthful_features, feature_cols)
    group_df = compute_group_summary(stats_df)

    print_feature_statistics(stats_df, top_k=top_k)
    print_group_summary(group_df)

    output_stats_path = OPENSMILE_DIR / "opensmile_feature_stats.csv"
    output_group_path = OPENSMILE_DIR / "opensmile_group_summary.csv"
    stats_df.to_csv(output_stats_path, index=False)
    group_df.to_csv(output_group_path, index=False)
    print(f"\nSaved feature statistics to: {output_stats_path}")
    print(f"Saved group summary to: {output_group_path}")

    if visualize:
        print("\n" + "=" * 100)
        print("Generating Visualizations")
        print("=" * 100)
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        print("  Plotting top feature comparison...")
        plot_top_feature_comparison(stats_df, PLOTS_DIR, top_k=min(top_k, 20))
        print("  Plotting group summary...")
        plot_group_summary(group_df, PLOTS_DIR)
        print(
            "  Plotting feature distributions "
            f"(max {distribution_max_points} frame rows per class/feature; 0 = all rows)..."
        )
        plot_feature_distribution(
            deceptive_features,
            truthful_features,
            stats_df,
            PLOTS_DIR,
            top_k=min(top_k, 10),
            max_points=distribution_max_points,
        )
        print(f"\n✓ All visualizations saved to: {PLOTS_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute openSMILE statistics from utterance-level CSV features"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of clips to process per class (for testing)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=TOP_K_DEFAULT,
        help="Number of top features to print / highlight",
    )
    parser.add_argument(
        "--distribution_max_points",
        type=int,
        default=MAX_DISTRIBUTION_POINTS,
        help=(
            "Maximum sampled frame rows per class/feature for the distribution plot. "
            "Use 0 to plot all rows."
        ),
    )
    args = parser.parse_args()
    main(
        limit=args.limit,
        visualize=args.visualize,
        top_k=args.top_k,
        distribution_max_points=args.distribution_max_points,
    )
