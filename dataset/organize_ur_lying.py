#!/usr/bin/env python3
"""Create symbolic links organizing UR_LYING dataset by Deceptive/Truthful.

Mirrors the Real-life_Deception_Detection_2016 structure:
    organized/
    ├── Clips/
    │   ├── Deceptive/   → symlinks to processed merged videos (W-B)
    │   └── Truthful/    → symlinks to processed merged videos (W-T)
    └── Transcription/
        ├── Deceptive/   → symlinks to commanded_low_stakes transcripts (W-B)
        └── Truthful/    → symlinks to commanded_low_stakes transcripts (W-T)

Classification is determined by matching each processed video to its raw W-file:
    - W-B (Bluff) → Deceptive
    - W-T (Truth) → Truthful
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent / "UR_LYING_Deception_Dataset"
PROCESSED_DIR = BASE_DIR / "v1.0" / "processed_data" / "merged_videos"
RAW_DIR = BASE_DIR / "v1.0" / "raw_data"
ORGANIZED_DIR = BASE_DIR / "organized"
SPLITS_DIR = BASE_DIR / "splits"

CONDITIONS = ["commanded_low_stakes", "commanded_med_stakes", "voluntary_med_stakes"]


def build_raw_index(raw_videos_dir: Path) -> dict[str, str]:
    """Build file_id → classification mapping from raw W-files."""
    index: dict[str, str] = {}
    for f in raw_videos_dir.glob("*.mp4"):
        m = re.search(r"-W-(B|T)-", f.name)
        if m:
            label = "Deceptive" if m.group(1) == "B" else "Truthful"
            # file_id = everything before "-W-B-" or "-W-T-"
            file_id = f.name[: f.name.index(f"-W-{m.group(1)}-")]
            index[file_id] = label
    return index


def get_processed_file_id(stem: str, condition: str) -> str:
    """Extract the raw data file_id from a processed video stem.

    commanded_low_stakes uses HH-MM-SS-ms naming; the raw file_id is the
    last 3 segments (MM-SS-ms). All other conditions use the full timestamp
    stem as the file_id.
    """
    if condition == "commanded_low_stakes":
        parts = stem.split("-")
        return "-".join(parts[1:])  # drop leading HH segment
    return stem


def main() -> None:
    for category in ("Deceptive", "Truthful"):
        (ORGANIZED_DIR / "Clips" / category).mkdir(parents=True, exist_ok=True)
        (ORGANIZED_DIR / "Transcription" / category).mkdir(parents=True, exist_ok=True)

    stats: dict[str, int] = {"Deceptive": 0, "Truthful": 0, "unmatched": 0}
    transcript_stats: dict[str, int] = {"Deceptive": 0, "Truthful": 0}

    for condition in CONDITIONS:
        processed_dir = PROCESSED_DIR / condition
        raw_video_dir = RAW_DIR / condition / "videos"
        transcript_dir = RAW_DIR / condition / "transcripts"

        raw_index = build_raw_index(raw_video_dir)

        for video in sorted(processed_dir.rglob("*.mp4")):
            file_id = get_processed_file_id(video.stem, condition)
            classification = raw_index.get(file_id)

            if classification is None:
                print(f"WARNING: No W-file match for {video.name} (file_id={file_id})")
                stats["unmatched"] += 1
                continue

            # Symlink for video
            clip_link = ORGANIZED_DIR / "Clips" / classification / video.name
            if not clip_link.exists():
                clip_link.symlink_to(video.resolve())
            stats[classification] += 1

            # Symlink for transcript (only commanded_low_stakes has transcripts)
            if transcript_dir.exists():
                transcript = transcript_dir / f"{file_id}.csv"
                if transcript.exists():
                    trans_link = (
                        ORGANIZED_DIR / "Transcription" / classification / f"{file_id}.csv"
                    )
                    if not trans_link.exists():
                        trans_link.symlink_to(transcript.resolve())
                    transcript_stats[classification] += 1

    total_clips = stats["Deceptive"] + stats["Truthful"]
    total_trans = transcript_stats["Deceptive"] + transcript_stats["Truthful"]
    print(
        f"Clips:         {stats['Deceptive']} Deceptive, {stats['Truthful']} Truthful "
        f"(total={total_clips}, unmatched={stats['unmatched']})"
    )
    print(
        f"Transcription: {transcript_stats['Deceptive']} Deceptive, "
        f"{transcript_stats['Truthful']} Truthful (total={total_trans})"
    )


SPLIT_SEED = 42


def _extract_trial_id(filename: str) -> str | None:
    """Extract trial_id = everything before the -I-/-W- role marker."""
    m = re.search(r"-(I|W)-(B|T)-", filename)
    return filename[: m.start()] if m else None


def _build_combined_trial_split(
    csv_paths: list[tuple[Path, str]], seed: int
) -> dict[str, dict[str, set[tuple[str, str]]]]:
    """Pass 1: read filename column only from all 3 CSVs, return train/test sets.

    Returns: {class: {"train": {(condition, trial_id), ...}, "test": {...}}}
    Only W-files (witness behavior) determine the split.
    """
    # Build a set of valid (condition, trial_id) pairs that have a processed video.
    # commanded_low_stakes clip stems are HH-MM-SS-ms; trial_id is the last 3 segments.
    valid_trials: set[tuple[str, str]] = set()
    for cls in ("Deceptive", "Truthful"):
        for clip in (ORGANIZED_DIR / "Clips" / cls).iterdir():
            stem = clip.stem
            parts = stem.split("-")
            if len(parts) == 4 and not stem[0].isalpha():
                # commanded_low_stakes: drop leading HH segment
                valid_trials.add(("commanded_low_stakes", "-".join(parts[1:])))
            else:
                # timestamp format: full stem is the trial_id
                for cond in ("commanded_med_stakes", "voluntary_med_stakes"):
                    valid_trials.add((cond, stem))

    trial_class: dict[tuple[str, str], str] = {}
    for csv_path, condition in csv_paths:
        filenames = pd.read_csv(csv_path, usecols=["filename"])["filename"].unique()
        for fname in filenames:
            if "-W-B-" in fname:
                cls = "Deceptive"
            elif "-W-T-" in fname:
                cls = "Truthful"
            else:
                continue
            tid = _extract_trial_id(fname)
            if tid and (condition, tid) in valid_trials:
                trial_class[(condition, tid)] = cls

    rng = np.random.default_rng(seed)
    split: dict[str, dict[str, set]] = {}
    for cls in ("Deceptive", "Truthful"):
        trials = np.array([(c, t) for (c, t), lbl in trial_class.items() if lbl == cls])
        rng.shuffle(trials)
        n_train = max(1, int(len(trials) * 0.9))
        split[cls] = {
            "train": {tuple(r) for r in trials[:n_train]},
            "test": {tuple(r) for r in trials[n_train:]},
        }
    return split


def _write_combined_csvs(
    csv_paths: list[tuple[Path, str]],
    split: dict,
    organized_dir: Path,
) -> None:
    """Pass 2: stream all condition CSVs, write one CSV per trial per folder.

    Output: organized_dir/{Train|Test}/{Deceptive|Truthful}/<trial>.csv
    W-only rows (Interrogator rows are excluded). The filename column is
    dropped so OpenFaceDataset can select feature columns directly.
    """
    lookup: dict[tuple[str, str], tuple[str, str]] = {}
    for cls in ("Deceptive", "Truthful"):
        for split_name in ("train", "test"):
            for cond, tid in split[cls][split_name]:
                lookup[(cond, tid)] = (split_name.capitalize(), cls)

    # Accumulate per-trial frames: {(split_name, cls, out_filename): [frames]}
    trial_buffers: dict[tuple[str, str, str], list[pd.DataFrame]] = {}

    for csv_path, condition in csv_paths:
        print(f"  Streaming {condition}...")
        for chunk in pd.read_csv(csv_path, chunksize=50_000):
            w_mask = chunk["filename"].str.contains(r"-W-[BT]-", regex=True)
            chunk = chunk[w_mask].copy()
            chunk["_tid"] = chunk["filename"].apply(_extract_trial_id)
            for tid, grp in chunk.groupby("_tid"):
                key = (condition, tid)
                if key not in lookup:
                    continue
                split_name, cls = lookup[key]
                # e.g. "36-04-870-W-B-user98_openface.csv" → "36-04-870-W-B-user98.csv"
                raw_fname = grp["filename"].iloc[0]
                out_name = raw_fname.replace("_openface.csv", ".csv")
                buf_key = (split_name, cls, out_name)
                trial_buffers.setdefault(buf_key, []).append(
                    grp.drop(columns=["_tid", "filename"])
                )

    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    for (split_name, cls, out_name), frames in trial_buffers.items():
        out_path = organized_dir / split_name / cls / out_name
        pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
        counts[(split_name, cls)] += 1

    for (split_name, cls), n in sorted(counts.items()):
        print(f"  Wrote {n} trial CSVs to {split_name}/{cls}/")


def split_openface_and_symlinks() -> None:
    """Split OpenFace data into 4 combined CSVs; create Train/Test symlink trees."""
    openface_src = BASE_DIR / "v1.0" / "processed_data" / "openface_features"

    csv_paths: list[tuple[Path, str]] = []
    for condition in CONDITIONS:
        p = openface_src / f"{condition}_openface.csv.bz2"
        if not p.exists():
            p = openface_src / f"{condition}_openface.csv"
        csv_paths.append((p, condition))

    for split_name in ("Train", "Test"):
        for category in ("Deceptive", "Truthful"):
            (SPLITS_DIR / split_name / category).mkdir(parents=True, exist_ok=True)
            (SPLITS_DIR / split_name / category / "Clips").mkdir(exist_ok=True)
            (SPLITS_DIR / split_name / category / "Transcription").mkdir(exist_ok=True)

    print("Pass 1: computing trial splits across all conditions...")
    split = _build_combined_trial_split(csv_paths, SPLIT_SEED)
    for cls in ("Deceptive", "Truthful"):
        print(f"  {cls}: {len(split[cls]['train'])} train, {len(split[cls]['test'])} test trials")

    print("\nPass 2: writing combined openface CSVs...")
    _write_combined_csvs(csv_paths, split, SPLITS_DIR)

    print("\nCreating Clips and Transcription symlinks...")
    for cls in ("Deceptive", "Truthful"):
        for split_name_lower in ("train", "test"):
            split_name = split_name_lower.capitalize()
            for condition, trial_id in split[cls][split_name_lower]:
                if condition == "commanded_low_stakes":
                    matches = list((ORGANIZED_DIR / "Clips" / cls).glob(f"*-{trial_id}.mp4"))
                else:
                    matches = list((ORGANIZED_DIR / "Clips" / cls).glob(f"{trial_id}.mp4"))
                for src in matches:
                    dst = SPLITS_DIR / split_name / cls / "Clips" / src.name
                    if not dst.exists():
                        dst.symlink_to(src.resolve())

                trans_src = ORGANIZED_DIR / "Transcription" / cls / f"{trial_id}.csv"
                if trans_src.exists():
                    dst = SPLITS_DIR / split_name / cls / "Transcription" / f"{trial_id}.csv"
                    if not dst.exists():
                        dst.symlink_to(trans_src.resolve())


if __name__ == "__main__":
    main()
    split_openface_and_symlinks()
